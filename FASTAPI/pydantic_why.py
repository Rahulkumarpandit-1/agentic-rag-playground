from pydantic import BaseModel,EmailStr,Field
from typing import List,Dict,Optional,Annotated

class Patient(BaseModel):
    name:Annotated[str,Field(max_length=50,title='name of the patient',Description='name of the patient',example='nitish')]
    age:int
    email:EmailStr
    weight:float
    married:bool=True
    allergies:Optional[List[str]]=Field(max_length=5)
    contact_details:Dict
    
    


def insert_patiet_data(patient:Patient):
    print(patient.name)
    print(patient.age) 
    print(patient.weight)
    print(patient.married)
    print(patient.email)
    print(patient.allergies)
    print(patient.contact_details)
    print("inserted")
      

patient_info={'name':'nitish','email':'zYJWd@example.com','age':30,'weight':80.0,'allergies':['penicillin','eggs','Dog'],'contact_details':{'phone':1234567890}}
patient1=Patient(**patient_info)

insert_patiet_data(patient1)