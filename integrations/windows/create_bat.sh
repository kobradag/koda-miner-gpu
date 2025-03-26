echo REM When mining to a local node, you can drop the -s option. > ${1}/mine.bat
echo echo ============================================================ >> ${1}/mine.bat
echo echo = Running Kobra Miner with default .bat. Edit to configure = >> ${1}/mine.bat
echo echo ============================================================ >> ${1}/mine.bat
echo :start >> ${1}/mine.bat
echo ${1}.exe -a qrzs2hd6rtcx2zd4dzkzrpqjx4jg8ndmqqjle8j9cpp93gg059tludxxvvfqd -s n.seeder1.kobra.net >> ${1}/mine.bat
echo goto start >> ${1}/mine.bat


# target\release\kobra-miner -a kobra:qz8wuydm0hjgxduj7x4fwslg2pvh4g2q96l83w84g2smc0gfmdd45kwue5ada -s localhost
