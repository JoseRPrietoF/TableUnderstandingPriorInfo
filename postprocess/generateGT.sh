for file in *.xml; do

awk '{
        if($0~"TableRegion")
                table=1;
                if((table==1)&&(($0~"<TableCell")||($0~"<Unicode")||($0~"TableRegion")))
                print $0
}' $file | awk '{
if(($0~"TableCell")||($0~"TableRegion")){
                print "";
                printf("%s", $0);
        }else
                printf("%s",$0);
}' | sed 's/<Unicode>//g' | sed 's/<\/Unicode>//g'|sed 's/ [ ]*/ /g' | awk 'BEGIN{n=0;}
{
   if($0~"<TableRegion"){n=n+1;};
   cell[n,$3,$2]=$0;
}
END{

    for(c=1;c<=15;c++){
         print cell[1,"col=\"0\"","row=\"0\""], cell[1,"col=\""c"\"","row=\"0\""],cell[1,"col=\""c"\"","row=\"1\""] >>  "col"c".txt"

         for(r=2;r<=15;r++){
                 if(cell[1,"col=\""c"\"","row=\""r"\""]!="") print cell[1,"col=\"0\"","row=\""r"\""], cell[1,"col=\""c"\"","row=\""r"\""] >>  "col"c".txt"
         }
         for(r=0;r<=15;r++){
            if(cell[2,"col=\""c"\"","row=\""r"\""]) print cell[2,"col=\"0\"","row=\""r"\""], cell[2,"col=\""c"\"","row=\""r"\""] >>  "col"c".txt"
          }
   }
}'


for f in col*; do sed 's/<[^>]*>//g' $f | awk '{if(NF>1) print $0}' > k; mv k $f; done

mv col1.txt knowts_${file/xml/txt}
mv col2.txt Fathoms_${file/xml/txt}
mv col3.txt Courses_${file/xml/txt}
mv col4.txt WindDirection_${file/xml/txt}
mv col5.txt WindForce_${file/xml/txt}
mv col6.txt Leeway_${file/xml/txt}
mv col7.txt BarometerHeight_${file/xml/txt}
mv col8.txt BarometerTher_${file/xml/txt}
mv col9.txt AirTemperature_${file/xml/txt}
mv col10.txt BulbTemperature_${file/xml/txt}
mv col11.txt SeaTemperature_${file/xml/txt}
mv col12.txt WeatherState_${file/xml/txt}
mv col13.txt Clouds_${file/xml/txt}
mv col14.txt ClearSky_${file/xml/txt}
mv col15.txt SeaState_${file/xml/txt}

done