rm a.txt b.txt
find -type l -name "*" >a.txt
for ln in `cat a.txt`; do
	echo `readlink $ln` >>b.txt
	echo $ln >>b.txt
done
var="link"
link=
for ln in `cat b.txt`; do
        if [ "$var" = "link" ]
	then
		link=$ln
		var="tar"
        elif [ "$var" = "tar" ]
	then
                echo "rm -rf $ln"
		echo "cp -rf $link $ln"
                rm -rf $ln
                cp -rf $link $ln
		var="link"
	fi		 
done
