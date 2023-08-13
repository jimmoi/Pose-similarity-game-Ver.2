วิธีใช้งาน
1. ติดตั้ง python 3.10.12 และ ติดตั้ง package ตาม requirement เข้า cmd "พิมพ์ pip install -r ที่อยู่ไฟลrequirement"
5.เปิด folder "ver.2 but 95% yet" ด้วย  VScode ทำการ trust folder
3. นำรูป reference ไปวางไว้ไน "\raw ref img\put image here"
4. ทำการ execute file ที่ชื่อว่า "Run this file"
5.เมื่อโปรแกรมทำงานแล้ว ต้องการไปข้อต่อไป กด n ต้องการออก กด q **ต้องเป็นภาษาอังกฤษ

การทำงานคราว
1.get_point_and drawed image.py 
	ทำการดึงรูปจาก folder แล้วทำการเก็บค่าไว้ที่ all_reference_poses_point_deletable.py
	และทำการวาด landmark ลงบนรูป และเก็บไว้ที่ folder "Drawed reference image deletable"
	
2.needed_func.py
	เก็บ function ที่จำเป็น ไว้เรียกใช้ 
	
3.Run_this_file
	ใช้เพื่อเริ่มทำงาน
	
4.config
	ใช้ optimizeต่างๆ
	round_poes < จำนวนรูป  	---> จำนวนด่าน
	image_scale 		--->แปลงความสูงรูป
	scale optimize 		--->ขนาดหน้าต่าง
	cam_type 		--->เปลี่ยนชนิดกล้อง แนวตั้ง="Hor" แนวนอน="Ver"
	crop_Hor		--->ตัดกรอบในแนวตั้ง
	crop_Ver		--->ตัดกรอบในแนวนอน
	window_name		--->ชื่อหน้าต่าง
