ffmpeg -y -nostdin -i $1 -an -vf scale=3840:2160 -c:v rawvideo -pix_fmt yuv420p10le $2 -loglevel panic