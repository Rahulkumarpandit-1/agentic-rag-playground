from pydantic import BaseModel,EmailStr,Field,field_validator,model_validator,computed_field
from typing import List,Dict,Optional,Annotated

class Patient(BaseModel):
    name:str
    age:int
    email:EmailStr
    weight:float
    height:float
    married:bool=True
    allergies:List[str]
    contact_details:Dict
    
    @field_validator('email')
    @classmethod
    def email_validator(cls,value):
     valid_domains=['hdfc.com','icicibank.com']    
     domain_name=value.split('@')[-1]
    
     if domain_name  not in valid_domains:
         raise ValueError("Not valid domain name ")
    
     return value
    
    @field_validator('name')
    @classmethod        
    def trandform_name(cls,value):
        return value.upper()
          
    @model_validator(mode='after')
    def validate_emergency_contact(cls,model):
        if model.age>60 and 'emergency' not in model.contact_details:
            raise ValueError('if patient age>60 ,it must have emergency contact')
        return model
    
    @computed_field
    @property
    def calculate_bmi(self)->float:
        bmi=round(self.weight/(self.height**2),2)   
        return bmi
    
def insert_patiet_data(patient:Patient):
    print(patient.name)
    print(patient.age) 
    print(patient.weight)
    print(patient.married)
    print(patient.email)
    print('bmi',patient.calculate_bmi)
    print(patient.allergies)
    print(patient.contact_details)
    print("inserted")
      
      

patient_info={'name':'nitish','email':'zYJWd@hdfc.com','height':1.65,'age':61,'weight':80.0,'allergies':['penicillin','eggs','Dog'],'contact_details':{'phone':1234567890,'emergency':'2444234'}}
patient1=Patient(**patient_info)

insert_patiet_data(patient1)