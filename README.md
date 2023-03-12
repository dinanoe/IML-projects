create python eviroment:

```bash
python -m venv venv
```

activate enviroment:

```bash
source venv/bin/activate #linux
venv\Scripts\activate.bat #Windows
```

install the dependecies

```bash
pip install -r requirements.txt
```

freeze dependencies

```bash
pip freeze > requirements.txt
```

If you have a free problem, open a terminal as adminstrator and run

```
netsh winsock reset
```
