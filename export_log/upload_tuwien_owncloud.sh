token=7EOrGo99qGVR4qA
psw=dreamer
logdir=${1}

echo -e "[Info] Creating tar ball for log dir ${logdir}\n"
tarball="$(basename -- ${logdir}).tar"
tar cvf ${tarball} --exclude='*/episodes' ${logdir}
echo -e "[Info] Created ${tarball}\n"

echo "[Info] Uploading ${tarball} ..."
time curl -u ${token}:${psw} -T ${tarball} "https://owncloud.tuwien.ac.at/public.php/webdav/${tarball}"
echo "[Info] Done."