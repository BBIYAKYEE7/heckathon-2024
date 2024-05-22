import turtle

# Create a turtle object
t = turtle.Turtle()

# Set the turtle color to red
t.color("red")

# Set the turtle fill color to red
t.begin_fill()

# Draw the left side of the heart
t.left(140)
t.forward(180)
t.circle(-90, 200)

# Move turtle to the right side
t.setheading(60)
t.circle(-90, 200)
t.forward(180)

# End the fill
t.end_fill()

# Hide the turtle
t.hideturtle()

# Keep the window open
turtle.done()