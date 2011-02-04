import hadoopy
import cv
import shutil
import os

def fetch_output():
    global faces_output
    faces_output = [x for x in hadoopy.cat('faces/face_output')]

def prepare_output():
    with open('face_output.html','w') as output:
        output.write(header)
        try:
            shutil.rmtree('faces/')
        except:
            pass
        os.mkdir('faces')
        for (filename,val) in faces_output:
            with open(filename,'w') as f:
                f.write(val)
            img = cv.LoadImage(filename)
            cv.ShowImage('face',img)
            cv.WaitKey(100)
                
        
        for (filename,val) in faces_output:
                if not filename == 'mean_face.jpg':            
                    output.write(template % dict(face_url=filename))
                else:
                    mean_face = val
        output.write(footer)
            
    
header = """
<div style="text-align:center">
<h3>The Average Face of George Bush</h3>
<img src="mean_face.jpg" style="width:128px;height:128px">
</div>
<h3>All the detected faces:</h3>
"""
footer = """
"""
template = r"""
<img src="%(face_url)s"/>
"""

if __name__ == "__main__":
    if not 'faces_output' in globals(): 
        print 'fetching the faces'
        fetch_output()
