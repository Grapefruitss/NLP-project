#!/usr/bin/env bash
# exit on error
set -o errexit

start_time=$(date +%s)

echo "Starting directory: $(pwd)"

STORAGE_DIR=/opt/render/project/.render
TEMP_DIR=/tmp/apt-lists

if [[ ! -d $STORAGE_DIR/chrome ]]; then
  echo "...Downloading Chrome"
  mkdir -p $STORAGE_DIR/chrome
  cd $STORAGE_DIR/chrome
  wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  
  # Set the TMPDIR environment variable
  export TMPDIR=$TEMP_DIR
  
  # Create the necessary directories with write permissions
  mkdir -p $TEMP_DIR/apt/lists/partial
  
  apt-get update -o Dir::State::lists="$TEMP_DIR/apt/lists" -o Dir::State::status="$TEMP_DIR/apt/status"
  apt-get install -y --no-install-recommends libgtk2.0-dev libgtk-3-dev libgbm-dev libx11-xcb-dev libxcomposite-dev \
                     libxcursor-dev libxdamage-dev libxi-dev libxrandr-dev libxss-dev libxtst-dev \
                     libatk1.0-dev libatk-bridge2.0-dev libpangocairo-1.0-0 libcups2-dev libfontconfig1-dev \
                     libdbus-1-dev libexpat1-dev
  dpkg -i google-chrome-stable_current_amd64.deb
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
export PATH="${PATH}:/opt/render/project/.render/chrome/opt/google/chrome"

# add your own build commands...
echo "Installing Python dependencies..."
pip install -r requirements.txt --no-cache-dir -q

echo "Starting application..."
uvicorn app:app --host 0.0.0.0 --port 8080 &

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Build completed in $elapsed_time seconds."
