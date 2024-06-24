#!/usr/bin/env bash
# exit on error
set -o errexit

STORAGE_DIR=/opt/render/project/.render

if [[ ! -d $STORAGE_DIR/chrome ]]; then
  echo "...Downloading Chrome"
  mkdir -p $STORAGE_DIR/chrome
  cd $STORAGE_DIR/chrome
  wget -P ./ https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  dpkg -x ./google-chrome-stable_current_amd64.deb $STORAGE_DIR/chrome
  rm ./google-chrome-stable_current_amd64.deb
  cd $HOME/project/src # Make sure we return to where we were
else
  echo "...Using Chrome from cache"
fi

# be sure to add Chrome's location to the PATH as part of your Start Command
export PATH="${PATH}:/opt/render/project/.render/chrome/opt/google/chrome"

# Check if requirements.txt exists in the expected directory
if [[ -f $HOME/project/requirements.txt ]]; then
  echo "Installing Python dependencies..."
  pip install -r $HOME/project/src/requirements.txt --no-cache-dir -q
else
  echo "ERROR: Could not find requirements.txt in the expected directory."
  exit 1
fi

echo "Starting application..."
uvicorn app:app --host 0.0.0.0 --port 8080 &

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Build completed in $elapsed_time seconds."
