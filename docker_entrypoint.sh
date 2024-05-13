#! /bin/sh

# Make sure storage folder path is set!
if [ -z "${STORAGE_FOLDER}" ]; then
    STORAGE_FOLDER="storage"
else
    STORAGE_FOLDER=${STORAGE_FOLDER_PATH}
fi

# Set environment variables, if needed
export UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
export UVICORN_PORT="${UVICORN_PORT:-3834}"

# Make sure logs folder exists
TIMESTAMP=$(date +"%Y_%m_%d_%H%M_%S")
FILENAME=$TIMESTAMP.txt
LOGDIR="$STORAGE_FOLDER/logs"
CLEANUPLOGDIR="$LOGDIR/cleanup_daemon"

# Create folders to hold log files, if needed
mkdir -p $CLEANUPLOGDIR
CLEANUPLOGFILE="$CLEANUPLOGDIR/$FILENAME"
echo ""
echo "Writing logs to:"
echo "$CLEANUPLOGFILE"
echo ""

# Run background cleanup script
python3 -u run_cleanup_daemon.py >> $CLEANUPLOGFILE 2>&1 &
PID1=$!
uvicorn run_todserver:asgi_app
PID2=$!

# On ctrl+c or kill events, be sure to kill ALL processes before closing!
close_all_on_exit()
{
	echo ""
	echo "Closing todserver..."
	kill $PID1
	kill $PID2
	exit 1
}
trap close_all_on_exit SIGINT SIGTERM

# Make sure all processes are shutdown
close_all_on_exit

