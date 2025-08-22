from flask import Flask,render_template, request, redirect,url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:///todo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']= False
db = SQLAlchemy(app)

class Todo(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    title = db.Column(db.String(200), nullable = False)
    desc = db.Column(db.String(500), nullable = False)
    date_created = db.Column(db.DateTime, default = lambda: datetime.now(timezone.utc))
    status = db.Column(db.Boolean, default = False)
        
    def __repr__(self):
        return f"{self.sno} - {self.title}"


@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form['title']
        description = request.form["description"]
        todo = Todo(title = title, desc = description)
        db.session.add(todo)
        db.session.commit()
        
    search_query = request.args.get('q')
    if search_query:
       alltodo = Todo.query.filter(Todo.title.contains(search_query)).all()
    else:
        alltodo = Todo.query.all()
    return render_template("index.html", alltodo = alltodo)

@app.route("/description/<int:sno>")
def describe(sno):
    todo=Todo.query.filter_by(sno=sno).first()
    return render_template("desc.html",todo = todo)

@app.route("/update/<int:sno>", methods = ['GET', 'POST'])
def update(sno):
    if request.method == 'POST':
        title = request.form['title']
        desc = request.form['description']
        todo = Todo.query.filter_by(sno=sno).first()
        todo.title = title
        todo.desc = desc
        db.session.add(todo)
        db.session.commit()
        return redirect(url_for("describe",sno=sno))
    todo = Todo.query.filter_by(sno=sno).first()
    return render_template("update.html", todo = todo)
    
    
@app.route("/toggle/<int:sno>")
def toggle_status(sno):
    todo = Todo.query.filter_by(sno=sno).first()
    todo.status = not todo.status
    db.session.commit()
    return redirect("/") 
    
@app.route("/delete/<int:sno>")
def delete(sno):
    todo = Todo.query.filter_by(sno=sno).first()
    db.session.delete(todo)
    db.session.commit()
    return redirect("/")



@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
    
    
