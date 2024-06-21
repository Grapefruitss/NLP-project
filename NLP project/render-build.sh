#!/usr/bin/env bash
# exit on error
set -o errexit

start_time=$(date +%s)

echo "Starting directory: $(pwd)"

STORAGE_DIR=/opt/render/project/.render

if [[ ! -d $STORAGE_DIR/chrome ]]; then
  echo "...Downloading Chrome"
  mkdir -p $STORAGE_DIR/chrome
  cd $STORAGE_DIR/chrome
  wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  dpkg -i google-chrome-stable_current_amd64.deb || sudo apt-get install -f -y
  rm google-chrome-stable_current_amd64.deb
  # Check if the directory exists before changing to it
  if [[ -d "$HOME/project/src" ]]; then
    cd $HOME/project/src
    echo "Changed directory to $HOME/project/src"
  else
    echo "Directory $HOME/project/src does not exist. Please check the path."
    exit 1
  fi
else
  echo "...Using Chrome from cache"
fi

# be sure to add Chrome's location to the PATH as part of your Start Command
# export PATH="${PATH}:/opt/render/project/.render/chrome/opt/google/chrome"

# Update package list to ensure everything is up-to-date
apt-get update

# add your own build commands...
echo "Installing Python dependencies..."
pip install -r requirements.txt --no-cache-dir -q

echo "Starting application..."
uvicorn app:app --host 0.0.0.0 --port 8000

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Build completed in $elapsed_time seconds."
