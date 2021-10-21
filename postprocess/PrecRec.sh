for f in *.txt; do
echo $f;
awk -v n=$f 'BEGIN{
        file="/data/HisClima/DatosHisclima/NER_vero/GT_generated/"n
        while(getline < file){
                if($0!~"Hour"){
                        if($0~"A. M.") t="M";
                        else if($0~"P. M.") t="T";
                        else{
                                h=$1;
                                $1="";
                                if($2=="\"") $2="";
                                gt[t,h]=$0;
                        }
                }

        }
}{
                if($0~"A. M."){
                        t="M";
                        h=$3;
                        $1="";
                        $2="";
                        $3="";
                        if($4=="\"") $4="";
                        gsub(" ","",$0);
                        gsub(" ","",gt[t,h]);
                        if($0==gt[t,h]) iguales="si"; else iguales="no"
                        print $0,gt[t,h],iguales;
                }
                else if($0~"P. M."){
                        t="T";
                        h=$3;
                        $1="";
                        $2="";
                        $3="";
                        if($4=="\"") $4="";
                        gsub(" ","",$0);
                        gsub(" ","",gt[t,h]);
                        if($0==gt[t,h]) iguales="si"; else iguales="no"
                        print $0,gt[t,h],iguales;
                }
                # else {
                #         iguales="mal"
                #         print $0,gt[t,h],iguales;
                # }
                # if (iguales == "no") {
                        # print $0,gt[t,h],iguales;
                # }     
}' $f
done \
> output
awk 'BEGIN{s=0;n=0;t=0}{
        if($NF=="si") s++;
        if($NF=="no") n++;
        t++;
       
}END{
  print s,n,3557,n+s;
  recall=s/3557;
  precision=s/(s+n);
  f1=(2*precision*recall)/(precision+recall);
  print precision, recall, f1;
}' output

# 3557
# 2496