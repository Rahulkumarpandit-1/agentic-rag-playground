from fastapi import FastAPI,Path,HTTPException,Query
import json
app=FastAPI()



def load_data():
    with open("FASTAPI/patient.json",'r')as f:
     data=json.load(f)
     
    return data
@app.get("/")
def hello():
    return {"message": "Patient management system"}

@app.get("/about")
def about():
    return{"message":"YA fully functional API to manage your patient record" }

@app.get('/view')
def view():
    data=load_data()
    
    return data

@app.get('/patient/{patient_id}')
def view_patient(patient_id:str=Path(...,description='ID of the patient in DB',example='P001')):
    data=load_data()
    
    if patient_id in data:
        return data[patient_id]
    raise HTTPException(status_code=404,detail="Patient not found")

@app.get('/sort')
def sort_patient(sort_by:str=Query(...,description='sort on the basis of height,weight or bmi'),
                                   order:str=Query('asc',description='sort in asc or desc order')):
    data=load_data()
    valid_field=['height','weight','bmi']
    
    if sort_by not in valid_field:
        raise HTTPException(status_code=400,detail="Invalid sort field SELECT from {valid_field}")
    if order not in ['asc','desc']:
        raise HTTPException(status_code=400,details='invalid order selected b/w asc and desc')
    
    
    sorted_data=sorted(data.values(),key=lambda x:x[sort_by],reverse=(order=='desc'))
    
    return sorted_data