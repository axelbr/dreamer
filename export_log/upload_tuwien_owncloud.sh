token=""
logdir=${1}
psw=""

echo -e "[Info] Creating tar ball for log dir ${logdir}\n"
tarball="$(basename -- ${logdir}).tar"
tar cvf ${tarball} --exclude='*/episodes' --exclude='*/checkpoints/*pkl' ${logdir}
echo -e "[Info] Created ${tarball}\n"

echo "[Info] Uploading ${tarball} ..."
time curl -u ${token}:${psw} -T ${tarball} "https://owncloud.tuwien.ac.at/public.php/webdav/${tarball}"
echo "[Info] Done."