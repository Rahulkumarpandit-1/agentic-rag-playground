from pydantic import BaseModel


class Address(BaseModel):
    city:str
    state:str
    pin:int 
class Patient(BaseModel):
    name:str
    gender:str
    age:int
    address:Address 
    
    
address_dict={'city':'Guwahati','state':'Assam','pin':123456}
address1=Address(**address_dict)

patient_dict={'name':'Ananya Sharma','gender':'female','age':28,'address':address1}

patient1=Patient(**patient_dict)    

print(patient1)