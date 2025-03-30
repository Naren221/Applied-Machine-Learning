from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/square", methods=["POST"])
def square():
    data = request.get_json()  # Get JSON data from the request
    number = data.get("number")  # Extract the number
    result = number ** 2  # Compute square
    return jsonify({"number": number, "square": result})  # Return JSON response

if __name__ == "__main__":
    app.run(debug=True)
