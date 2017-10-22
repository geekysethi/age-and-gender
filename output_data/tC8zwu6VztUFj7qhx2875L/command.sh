cp -r /code/* /output
cd /output
echo "[program:tensorboard]" > /etc/supervisor/conf.d/tensorboard.conf
echo "command=tensorboard --port 6001 --logdir=/output" >> /etc/supervisor/conf.d/tensorboard.conf
echo "autorestart=true" >> /etc/supervisor/conf.d/tensorboard.conf
echo "startretries=10000" >> /etc/supervisor/conf.d/tensorboard.conf
service supervisor start || true; /run_jupyter.sh --no-browser --NotebookApp.base_url='/notebooks/T3mKRVFKDymxcX7Hxaa2K6' --NotebookApp.token='' --NotebookApp.allow_origin='*' --NotebookApp.tornado_settings="{'headers': {'Content-Security-Policy': \"frame-ancestors 'self' www.floydhub.com \"}}" --NotebookApp.iopub_data_rate_limit=1.0e10