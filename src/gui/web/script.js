// Onclick of the button
document.querySelector("#b1").onclick = function () {
    // Call python's random_python function
    eel.do_stuff()(function (number) {
        // Update the div with a random number returned by python
        document.querySelector(".random_number").innerHTML = number;
    });
};

document.querySelector("#b2").onclick = function () {
    // Call python's random_python function
    eel.data_manager_do_stuff()(function (number) {
        // Update the div with a random number returned by python
        document.querySelector(".random_number").innerHTML = number;
    });
};
