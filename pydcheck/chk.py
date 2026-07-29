import pydantic
print('pydantic.__file__ =', getattr(pydantic, '__file__', '?'))
print('has BaseModel     =', hasattr(pydantic, 'BaseModel'))
try:
    class M(pydantic.BaseModel):
        x: int
    print('subclass OK       =', M(x=1))
except Exception as e:
    print('subclass FAILED   =', type(e).__name__, e)
