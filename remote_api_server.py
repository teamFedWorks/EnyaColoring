#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remote API Server for Video Pipeline
This server runs inside Docker containers and handles job execution.
Install: pip install flask flask-cors
Run: python remote_api_server.py --hoot 0.0.0.0 --
port 9090
"""

import os
import sys
import subprocess
import threading
import json
import time
import shutil
import urllib.request
import tempfile
import zipfile
import re
from urllib.parse import urlparse
from pathlib import Path
from flask import Flask, request, jsonify, send_file, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import argparse

app = Flask(__name__)
CORS(app)

# Configuration
WORKSPACE_DIR = '/workspace'
INPUT_VIDEOS_DIR = os.path.join(WORKSPACE_DIR, 'input_videos')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
BATCH_SCRIPT = 'youtube_urls_batch_run.py'

# Job status storage (in-memory, can be replaced with database)
jobs = {}
job_processes = {}
job_lock = threading.Lock()


def terminate_job_process(job_id):
    """Terminate the running subprocess for a job, if any."""
    process = job_processes.get(job_id)
    if not process:
        return False

    if process.poll() is not None:
        job_processes.pop(job_id, None)
        return False

    print(f"[API] Terminating process {process.pid} for job {job_id}")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"[API] Process {process.pid} did not terminate gracefully, killing...")
        process.kill()
    finally:
        job_processes.pop(job_id, None)
    return True


# Ensure directories exist
os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_input_from_url(file_url):
    """Download remote input to the local workspace."""
    try:
        parsed = urlparse(file_url)
        original_name = os.path.basename(parsed.path) or 'remote_input.mp4'
        safe_name = secure_filename(original_name)
        target_name = f"remote_{int(time.time() * 1000)}_{safe_name}"
        destination = os.path.join(INPUT_VIDEOS_DIR, target_name)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        print(f"[API] Downloading remote input from {file_url} -> {destination}")
        with urllib.request.urlopen(file_url) as response, open(destination, 'wb') as output_file:
            shutil.copyfileobj(response, output_file)
        return destination
    except Exception as exc:
        print(f"[API] Failed to download input URL {file_url}: {exc}")
        raise


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'workspace': WORKSPACE_DIR,
        'input_videos': INPUT_VIDEOS_DIR,
        'python_version': sys.version,
        'timestamp': time.time()
    })


@app.route('/status', methods=['GET'])
def status():
    """Get server status and workspace info"""
    workspace_exists = os.path.exists(WORKSPACE_DIR)
    input_videos_exists = os.path.exists(INPUT_VIDEOS_DIR)
    pipeline_exists = os.path.exists(os.path.join(WORKSPACE_DIR, 'pipeline.py'))
    wrapper_exists = os.path.exists(os.path.join(WORKSPACE_DIR, 'pipeline_wrapper.py'))
    
    # Check Python
    try:
        python_version = subprocess.check_output(['python', '--version'], stderr=subprocess.STDOUT).decode().strip()
    except:
        python_version = 'Unknown'
    
    return jsonify({
        'status': 'online',
        'workspace': {
            'path': WORKSPACE_DIR,
            'exists': workspace_exists
        },
        'input_videos': {
            'path': INPUT_VIDEOS_DIR,
            'exists': input_videos_exists
        },
        'files': {
            'pipeline_py': pipeline_exists,
            'pipeline_wrapper_py': wrapper_exists
        },
        'python': python_version,
        'timestamp': time.time()
    })


@app.route('/gpu/status', methods=['GET'])
def gpu_status():
    """Get GPU status using nvidia-smi"""
    try:
        # Run nvidia-smi query for GPU utilization and memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return jsonify({
                'error': 'nvidia-smi failed',
                'message': result.stderr
            }), 500
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'utilization_gpu': int(parts[2]),
                    'memory_used_mb': int(parts[3]),
                    'memory_total_mb': int(parts[4]),
                    'temperature': int(parts[5]),
                    'memory_used_percent': round((int(parts[3]) / int(parts[4])) * 100, 1) if int(parts[4]) > 0 else 0
                })
        
        return jsonify({
            'gpus': gpus,
            'timestamp': time.time()
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'error': 'nvidia-smi timeout',
            'message': 'GPU query took too long'
        }), 500
    except FileNotFoundError:
        return jsonify({
            'error': 'nvidia-smi not found',
            'message': 'nvidia-smi command is not available'
        }), 503
    except Exception as e:
        return jsonify({
            'error': 'GPU status error',
            'message': str(e)
        }), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload video file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(INPUT_VIDEOS_DIR, filename)
        file.save(filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': f'input_videos/{filename}',
            'full_path': filepath,
            'size': file_size,
            'message': 'File uploaded successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    with job_lock:
        return jsonify({
            'jobs': list(jobs.values()),
            'count': len(jobs)
        })


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job status"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(jobs[job_id])


@app.route('/jobs/<job_id>/batch_download', methods=['GET'])
def download_batch_zip(job_id):
    """Download all outputs for a batch job as a zip."""
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    job_type = job.get('job_type') or job.get('type')
    if job_type != 'batch_csv':
        return jsonify({'error': 'Batch download only available for batch jobs'}), 400

    batch_results = job.get('batch_results') or []
    files_to_zip = []

    for result in batch_results:
        final_output = result.get('finalOutput') or result.get('final_output')
        if not final_output:
            continue
        resolved = final_output
        if not os.path.isabs(resolved):
            resolved = os.path.join(WORKSPACE_DIR, resolved.lstrip('/\\'))
        resolved = os.path.normpath(resolved)
        if not resolved.startswith(os.path.normpath(WORKSPACE_DIR)):
            continue
        if os.path.isfile(resolved):
            video_url = result.get('videoUrl') or result.get('video_url') or ''
            arcname = safe_zip_name(video_url, os.path.basename(resolved))
            files_to_zip.append((resolved, arcname))

    if not files_to_zip:
        return jsonify({'error': 'No output files found for this batch'}), 404

    fd, temp_zip_path = tempfile.mkstemp(suffix='.zip')
    os.close(fd)

    try:
        with zipfile.ZipFile(temp_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in files_to_zip:
                zipf.write(file_path, arcname)

        @after_this_request
        def cleanup(response):
            try:
                os.remove(temp_zip_path)
            except OSError:
                pass
            return response

        return send_file(
            temp_zip_path,
            as_attachment=True,
            download_name=f'{job_id}_batch_outputs.zip',
            mimetype='application/zip'
        )
    except Exception as exc:
        try:
            os.remove(temp_zip_path)
        except OSError:
            pass
        print(f"[API] Failed to prepare batch zip for job {job_id}: {exc}")
        return jsonify({'error': 'Failed to prepare batch download'}), 500


@app.route('/batch_csv/upload', methods=['POST'])
def upload_batch_csv():
    """Upload or replace the master batch CSV file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Filename is required'}), 400

    target_path = request.form.get('targetPath') or os.environ.get('MASTER_CSV_PATH') or 'first_youtube_urls.csv'
    
    # If target_path doesn't specify batch_uploads, use batch_uploads folder
    if 'batch_uploads' not in target_path and not target_path.startswith('/'):
        # Default to batch_uploads folder for uploaded files
        filename = os.path.basename(target_path)
        target_path = f'batch_uploads/{filename}'
    
    resolved_path = target_path
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(WORKSPACE_DIR, resolved_path.strip('/\\'))
    resolved_path = os.path.normpath(resolved_path)

    if not resolved_path.startswith(os.path.normpath(WORKSPACE_DIR)):
        return jsonify({'error': 'Invalid target path'}), 400

    # Ensure directory exists
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    file.save(resolved_path)

    relative_path = os.path.relpath(resolved_path, WORKSPACE_DIR).replace('\\', '/')
    print(f"[API] Batch CSV uploaded to {resolved_path}")
    print(f"[API] Relative path: {relative_path}")
    print(f"[API] File size: {os.path.getsize(resolved_path)} bytes")

    return jsonify({
        'success': True,
        'path': f'/{relative_path}',
        'full_path': resolved_path,
    })


@app.route('/jobs', methods=['POST'])
def create_job():
    """Create and execute a new job"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        job_id = f"job_{int(time.time() * 1000)}"
        
        job_type = data.get('type') or data.get('jobType') or data.get('mode')
        batch_csv_path = data.get('batchCsvPath') or data.get('batch_csv_path')

        job = {
            'id': job_id,
            'status': 'pending',
            'progress': 0,
            'input_method': data.get('inputMethod') or data.get('input_method'),
            'youtube_url': data.get('youtubeUrl') or data.get('youtube_url'),
            'manual_path': data.get('manualPath') or data.get('manual_path'),
            'input_url': data.get('inputUrl') or data.get('input_url'),
            'part_index': data.get('partIndex') or data.get('part_index'),
            'part_total': data.get('partTotal') or data.get('part_total'),
            'unet_flag': data.get('unetFlag') if 'unetFlag' in data else data.get('unet_flag', False),
            'face_restore_flag': data.get('faceRestoreFlag') if 'faceRestoreFlag' in data else data.get('face_restore_flag', False),
            'upscale_flag': data.get('upscaleFlag') if 'upscaleFlag' in data else data.get('upscale_flag', False),
            'upscale_value': float(data.get('upscaleValue') if 'upscaleValue' in data else data.get('upscale_value', 2.0)),
            'clahe_flag': data.get('claheFlag') if 'claheFlag' in data else data.get('clahe_flag', False),
            'job_type': job_type,
            'batch_csv_path': batch_csv_path,
            'created_at': time.time(),
            'updated_at': time.time(),
            'output': '',
            'output_path': None,
            'error': None,
            'batch_results': [],
            'processed_count': 0,
        }
        
        print(f"[API] Creating job {job_id} with data: {job}")
        
        with job_lock:
            jobs[job_id] = job
        
        # Execute job in background
        thread = threading.Thread(target=execute_job, args=(job_id, job))
        thread.daemon = True
        thread.start()
        
        return jsonify(job), 201
    except Exception as e:
        print(f"[API] Error creating job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def execute_job(job_id, job):
    """Execute pipeline job"""
    job_type = job.get('job_type') or job.get('type')
    if job_type == 'batch_csv':
        execute_batch_job(job_id, job)
        return

    execute_standard_job(job_id, job)


def execute_standard_job(job_id, job):
    try:
        print(f"[API] Starting job execution: {job_id}")
        with job_lock:
            jobs[job_id]['status'] = 'running'
            jobs[job_id]['progress'] = 5
            jobs[job_id]['updated_at'] = time.time()
        
        manual_path = job.get('manual_path') or job.get('manualPath')
        if (not manual_path) and job.get('input_url'):
            downloaded_path = download_input_from_url(job.get('input_url'))
            manual_path = downloaded_path
            with job_lock:
                jobs[job_id]['manual_path'] = downloaded_path
        if manual_path:
            job['manual_path'] = manual_path
            
            # Verify the file exists before starting
            resolved_path = manual_path
            if not resolved_path.startswith('/'):
                resolved_path = os.path.join(WORKSPACE_DIR, resolved_path)
            resolved_path = os.path.normpath(resolved_path)
            
            print(f"[API] Resolved manual_path: {resolved_path}")
            if not os.path.exists(resolved_path):
                error_msg = f"Input file not found: {resolved_path} (original path: {manual_path})"
                print(f"[API] ERROR: {error_msg}")
                print(f"[API] WORKSPACE_DIR: {WORKSPACE_DIR}")
                print(f"[API] INPUT_VIDEOS_DIR: {INPUT_VIDEOS_DIR}")
                print(f"[API] Files in INPUT_VIDEOS_DIR: {os.listdir(INPUT_VIDEOS_DIR) if os.path.exists(INPUT_VIDEOS_DIR) else 'Directory does not exist'}")
                with job_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = error_msg
                    jobs[job_id]['updated_at'] = time.time()
                return
            else:
                file_size = os.path.getsize(resolved_path)
                print(f"[API] ✅ Input file found: {resolved_path} ({file_size / 1024 / 1024:.2f}MB)")

        # Build command
        command = build_pipeline_command(job)
        
        # Change to workspace directory
        original_cwd = os.getcwd()
        os.chdir(WORKSPACE_DIR)
        print(f"[API] Changed to workspace: {WORKSPACE_DIR}")
        
        # Execute pipeline
        print(f"[API] Executing command: {command}")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=WORKSPACE_DIR
        )
        with job_lock:
            jobs[job_id]['pid'] = process.pid
            jobs[job_id]['updated_at'] = time.time()
        job_processes[job_id] = process
        
        output_lines = []
        last_progress_update = time.time()
        
        for line in process.stdout:
            output_lines.append(line)
            # Print to terminal AND store in job output
            print(f"[Job {job_id}] {line.rstrip()}", flush=True)
            sys.stdout.flush()  # Ensure immediate output
            
            # Update logs incrementally for live streaming
            with job_lock:
                jobs[job_id]['output'] = ''.join(output_lines)
                jobs[job_id]['updated_at'] = time.time()
            
            # Update progress (simple parsing)
            progress = parse_progress(line)
            current_time = time.time()
            if progress is not None:
                with job_lock:
                    jobs[job_id]['progress'] = min(progress, 95)  # Keep at 95% until complete
                    jobs[job_id]['updated_at'] = current_time
                    last_progress_update = current_time
            elif current_time - last_progress_update > 10:
                # Increment progress slowly if no progress detected
                with job_lock:
                    current_progress = jobs[job_id].get('progress', 5)
                    if current_progress < 90:
                        jobs[job_id]['progress'] = min(current_progress + 1, 90)
                        jobs[job_id]['updated_at'] = current_time
                        last_progress_update = current_time
        
        process.wait()
        os.chdir(original_cwd)
        
        with job_lock:
            jobs[job_id]['output'] = ''.join(output_lines)
            jobs[job_id]['updated_at'] = time.time()
            
            jobs[job_id].pop('pid', None)
            job_processes.pop(job_id, None)

            if process.returncode == 0:
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['progress'] = 100
                jobs[job_id]['output_path'] = extract_output_path(output_lines)
                print(f"[API] Job {job_id} completed successfully")
            else:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = f'Pipeline failed with exit code {process.returncode}'
                print(f"[API] Job {job_id} failed with exit code {process.returncode}")
                
    except Exception as e:
        print(f"[API] Error executing job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        with job_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['updated_at'] = time.time()
            jobs[job_id].pop('pid', None)
        job_processes.pop(job_id, None)


def build_pipeline_command(job):
    """Build pipeline command for standard (non-batch) jobs"""
    job_type = job.get('job_type') or job.get('type')
    if job_type == 'batch_csv':
        raise ValueError('build_pipeline_command should not be called for batch CSV jobs. Use execute_batch_job instead.')
    
    parts = ['python', 'pipeline_wrapper.py']
    
    input_method = job.get('input_method', 'manual')
    
    # Check if input is a CSV file (should not happen for standard jobs)
    manual_path = job.get('manual_path') or job.get('manualPath')
    if manual_path and (manual_path.lower().endswith('.csv') or '/batch' in manual_path.lower()):
        raise ValueError(f'CSV files are not supported by pipeline_wrapper.py. Received: {manual_path}. Use youtube_urls_batch_run.py for batch processing.')
    
    if input_method == 'youtube':
        youtube_url = job.get('youtube_url') or job.get('youtubeUrl')
        if youtube_url:
            parts.append(f'"{youtube_url}"')
        else:
            raise ValueError('YouTube URL is required for youtube input method')
    else:
        # Resolve manual path
        if not manual_path:
            raise ValueError('Manual path is required for manual input method')
        
        if not manual_path.startswith('/'):
            manual_path = os.path.join(WORKSPACE_DIR, manual_path)
        parts.append(manual_path)
    
    # Add flags (handle both camelCase and snake_case)
    parts.append('true' if job.get('unet_flag') or job.get('unetFlag') else 'false')
    parts.append('true' if job.get('face_restore_flag') or job.get('faceRestoreFlag') else 'false')
    parts.append('true' if job.get('upscale_flag') or job.get('upscaleFlag') else 'false')
    parts.append(str(job.get('upscale_value') or job.get('upscaleValue', 2.0)))
    parts.append('true' if job.get('clahe_flag') or job.get('claheFlag') else 'false')
    
    # Add quality mode (quality or fast)
    quality_mode = job.get('quality_mode') or job.get('qualityMode') or 'quality'
    parts.append(quality_mode)
    
    command = ' '.join(parts)
    print(f"[API] Built command: {command}")
    print(f"[API] Quality mode: {quality_mode} → Will use {'pipeline.py' if quality_mode == 'quality' else 'pipeline_colorful.py'}")
    return command


def parse_progress(line):
    """Parse progress from output line"""
    # Look for progress patterns
    import re
    progress_match = re.search(r'Progress:\s*(\d+)%', line, re.IGNORECASE)
    if progress_match:
        return int(progress_match.group(1))
    
    # Look for task progress
    task_match = re.search(r'Task\s+(\d+)\s*/\s*(\d+)', line, re.IGNORECASE)
    if task_match:
        current = int(task_match.group(1))
        total = int(task_match.group(2))
        return int((current / total) * 100)
    
    return None


def extract_output_path(output_lines):
    """Find the final video path from the pipeline output."""
    import re
    for line in reversed(output_lines):
        match = re.search(r'final\s+video\s+at:\s*(.+)$', line, re.IGNORECASE)
        if match:
            resolved = match.group(1).strip()
            if not resolved:
                continue
            if not resolved.startswith('/'):
                resolved = os.path.join(WORKSPACE_DIR, resolved)
            normalized = os.path.normpath(resolved)
            print(f"[API] Detected output path: {normalized}")
            return normalized
    return None


def extract_batch_results(output_lines):
    """Parse per-video summaries from batch runner output."""
    results = []
    current = None

    for raw_line in output_lines:
        line = raw_line.strip()
        if not line:
            continue

        if 'Video URL:' in line:
            url = line.split('Video URL:')[-1].strip()
            current = {
                'videoUrl': url,
                'status': 'running',
                'log': [],
            }
            results.append(current)

        if not current:
            continue

        current['log'].append(raw_line.rstrip('\n'))

        normalized = line.lower()
        if normalized.startswith('📥 resolved path:') or 'Resolved Path:' in line:
            current['inputPath'] = line.split(':', 1)[1].strip()
        elif normalized.startswith('final video at:'):
            current['finalOutput'] = line.split(':', 1)[1].strip()
            current['status'] = 'completed'
        elif normalized.startswith('status:'):
            current['status'] = line.split(':', 1)[1].strip()
        elif normalized.startswith('remarks:'):
            current['remarks'] = line.split(':', 1)[1].strip()

    return results


def safe_zip_name(video_url, original_name):
    """Generate a safe filename for zip entries."""
    base = video_url or original_name
    if not base:
        base = original_name or 'video'
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', base)
    slug = slug.strip('_') or 'video'
    if len(slug) > 80:
        slug = slug[:80]
    return f"{slug}_{original_name}"


@app.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a job"""
    with job_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

        if jobs[job_id]['status'] in ['completed', 'failed', 'cancelled']:
            return jsonify({'error': 'Job cannot be cancelled'}), 400

    was_running = terminate_job_process(job_id)

    with job_lock:
        jobs[job_id]['status'] = 'cancelled'
        jobs[job_id]['progress'] = 0
        jobs[job_id]['updated_at'] = time.time()
        jobs[job_id].pop('pid', None)
        if was_running:
            jobs[job_id]['output'] = (jobs[job_id].get('output') or '') + '\n[API] Job cancelled by user.'

        return jsonify(jobs[job_id])


def execute_batch_job(job_id, job):
    """Execute batch CSV pipeline"""
    output_lines = []
    try:
        print(f"[API] Starting batch CSV job execution: {job_id}")
        with job_lock:
            jobs[job_id]['status'] = 'running'
            jobs[job_id]['progress'] = 5
            jobs[job_id]['updated_at'] = time.time()

        csv_path = (
            job.get('batch_csv_path')
            or job.get('batchCsvPath')
            or job.get('csv_path')
            or job.get('csvPath')
        )

        if not csv_path:
            raise ValueError('CSV path is required for batch jobs')

        print(f"[API][Batch] Original CSV path from job: {csv_path}")
        
        # Resolve CSV path - handle both absolute and relative paths
        resolved_csv = csv_path
        if not os.path.isabs(resolved_csv):
            # Remove leading slash if present for relative paths
            clean_path = resolved_csv.lstrip('/\\')
            resolved_csv = os.path.join(WORKSPACE_DIR, clean_path)
        
        resolved_csv = os.path.normpath(resolved_csv)
        
        # Also try with batch_uploads prefix if path doesn't exist
        if not os.path.exists(resolved_csv):
            # Try batch_uploads folder
            batch_uploads_path = os.path.join(WORKSPACE_DIR, 'batch_uploads', os.path.basename(resolved_csv))
            if os.path.exists(batch_uploads_path):
                print(f"[API][Batch] CSV not found at {resolved_csv}, found in batch_uploads: {batch_uploads_path}")
                resolved_csv = batch_uploads_path
            else:
                # Try original path with batch_uploads prefix
                if 'batch_uploads' not in resolved_csv:
                    alt_path = os.path.join(WORKSPACE_DIR, 'batch_uploads', os.path.basename(csv_path))
                    if os.path.exists(alt_path):
                        print(f"[API][Batch] CSV found in batch_uploads: {alt_path}")
                        resolved_csv = alt_path

        if not resolved_csv.startswith(WORKSPACE_DIR):
            raise ValueError(f'CSV path must be inside {WORKSPACE_DIR}. Got: {resolved_csv}')

        if not os.path.exists(resolved_csv):
            # List files in batch_uploads for debugging
            batch_uploads_dir = os.path.join(WORKSPACE_DIR, 'batch_uploads')
            if os.path.exists(batch_uploads_dir):
                files_in_batch_uploads = os.listdir(batch_uploads_dir)
                print(f"[API][Batch] Files in batch_uploads: {files_in_batch_uploads}")
            raise FileNotFoundError(f'CSV file not found: {resolved_csv} (original: {csv_path})')

        print(f"[API][Batch] ✅ CSV resolved path: {resolved_csv}")
        print(f"[API][Batch] CSV file exists: {os.path.exists(resolved_csv)}")
        if os.path.exists(resolved_csv):
            print(f"[API][Batch] CSV file size: {os.path.getsize(resolved_csv)} bytes")
            # Read first few lines to verify it's a valid CSV
            try:
                with open(resolved_csv, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline().strip() for _ in range(3)]
                    print(f"[API][Batch] CSV first line (header): {first_lines[0][:100] if first_lines else 'empty'}")
            except Exception as e:
                print(f"[API][Batch] Warning: Could not read CSV file: {e}")
        else:
            # List available files for debugging
            batch_uploads_dir = os.path.join(WORKSPACE_DIR, 'batch_uploads')
            if os.path.exists(batch_uploads_dir):
                available_files = os.listdir(batch_uploads_dir)
                print(f"[API][Batch] Available files in batch_uploads: {available_files}")
            workspace_files = [f for f in os.listdir(WORKSPACE_DIR) if f.endswith('.csv')]
            if workspace_files:
                print(f"[API][Batch] CSV files in workspace root: {workspace_files}")

        command = f'python {BATCH_SCRIPT}'

        original_cwd = os.getcwd()
        os.chdir(WORKSPACE_DIR)
        print(f"[API][Batch] Changed to workspace: {WORKSPACE_DIR}")
        print(f"[API][Batch] Executing command: {command}")

        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        env['MASTER_CSV_PATH'] = resolved_csv

        stdin_sequence = [
            'y',  # Automated mode
            'n',  # Disable per-video email notifications
            'n',  # Use alternate CSV path
            resolved_csv,  # Provide absolute CSV path
        ]

        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=WORKSPACE_DIR,
            env=env,
        )

        with job_lock:
            jobs[job_id]['pid'] = process.pid
            jobs[job_id]['updated_at'] = time.time()
            jobs[job_id]['batch_csv_path'] = resolved_csv
        job_processes[job_id] = process

        def feed_inputs():
            try:
                for response in stdin_sequence:
                    if process.poll() is not None:
                        break
                    payload = f"{response}\n"
                    process.stdin.write(payload)
                    process.stdin.flush()
                    print(f"[API][Batch] Sent response: {response}")
                    time.sleep(0.2)
            except Exception as stdin_error:
                print(f"[API][Batch] Failed to send stdin payload: {stdin_error}")
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        feeder = threading.Thread(target=feed_inputs, daemon=True)
        feeder.start()

        last_progress_update = time.time()

        for line in process.stdout:
            output_lines.append(line)
            # Print to terminal AND store in job output
            print(f"[Batch Job {job_id}] {line.rstrip()}", flush=True)
            sys.stdout.flush()  # Ensure immediate output
            
            # Update logs incrementally for live streaming
            with job_lock:
                jobs[job_id]['output'] = ''.join(output_lines)
                jobs[job_id]['updated_at'] = time.time()

            progress = parse_progress(line)
            current_time = time.time()
            if progress is not None:
                with job_lock:
                    jobs[job_id]['progress'] = min(progress, 95)
                    jobs[job_id]['updated_at'] = current_time
                    last_progress_update = current_time
            elif current_time - last_progress_update > 15:
                with job_lock:
                    current_progress = jobs[job_id].get('progress', 5)
                    if current_progress < 90:
                        jobs[job_id]['progress'] = min(current_progress + 1, 90)
                        jobs[job_id]['updated_at'] = current_time
                        last_progress_update = current_time

        process.wait()
        os.chdir(original_cwd)

        with job_lock:
            jobs[job_id]['output'] = ''.join(output_lines)
            jobs[job_id]['updated_at'] = time.time()
            jobs[job_id].pop('pid', None)
            jobs[job_id]['batch_results'] = extract_batch_results(output_lines)
            jobs[job_id]['processed_count'] = len(jobs[job_id]['batch_results'])
        job_processes.pop(job_id, None)

        with job_lock:
            if process.returncode == 0:
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['progress'] = 100
                jobs[job_id]['output_path'] = extract_output_path(output_lines)
                print(f"[API][Batch] Job {job_id} completed successfully")
            else:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = f'Batch pipeline failed with exit code {process.returncode}'
                print(f"[API][Batch] Job {job_id} failed with exit code {process.returncode}")

    except Exception as e:
        print(f"[API][Batch] Error executing job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        with job_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['updated_at'] = time.time()
            jobs[job_id].pop('pid', None)
            if output_lines:
                jobs[job_id]['output'] = ''.join(output_lines)
                jobs[job_id]['batch_results'] = extract_batch_results(output_lines)
        job_processes.pop(job_id, None)


@app.route('/files', methods=['GET'])
def list_files():
    """List files in input_videos directory"""
    try:
        files = []
        if os.path.exists(INPUT_VIDEOS_DIR):
            for filename in os.listdir(INPUT_VIDEOS_DIR):
                filepath = os.path.join(INPUT_VIDEOS_DIR, filename)
                if os.path.isfile(filepath):
                    files.append({
                        'name': filename,
                        'path': f'input_videos/{filename}',
                        'size': os.path.getsize(filepath),
                        'modified': os.path.getmtime(filepath)
                    })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download', methods=['GET'])
def download_file():
    """Download a file from the workspace"""
    try:
        relative_path = request.args.get('path')
        if not relative_path:
            return jsonify({'error': 'Path parameter is required'}), 400

        requested_path = os.path.normpath(relative_path)

        if os.path.isabs(requested_path):
            full_path = os.path.normpath(requested_path)
        else:
            normalized_path = requested_path.lstrip('/\\')
            full_path = os.path.normpath(os.path.join(WORKSPACE_DIR, normalized_path))

        if not full_path.startswith(os.path.normpath(WORKSPACE_DIR)):
            return jsonify({'error': 'Invalid path'}), 400

        if not os.path.isfile(full_path):
            return jsonify({'error': 'File not found'}), 404

        filename = os.path.basename(full_path)
        return send_file(full_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote API Server for Video Pipeline')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    print(f"Starting Remote API Server on {args.host}:{args.port}")
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Input Videos: {INPUT_VIDEOS_DIR}")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
