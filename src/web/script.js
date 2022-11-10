const createFeatureSelections = () => {
    let container = document.getElementById("featuresContainer");
    container.innerHTML = "";

    eel.get_xlabels()((features) => {
        // Create a checkbox for each feature
        for (let i = 0; i < features.length; i++) {
            let feature = features[i];
            let checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.id = "feature" + i;
            checkbox.checked = true;
            container.appendChild(checkbox);
            container.appendChild(document.createTextNode(feature));
            container.appendChild(document.createElement("br"));
        }
    });
};

document.getElementById("datasetFile").onclick = () => {
    eel.browse_files()(function (path) {
        document.getElementById("datasetPath").innerHTML = path || "No file selected";
        createFeatureSelections();
    });
};

document.getElementById("selectFeature").onclick = () => {
    let checkboxes = document.querySelectorAll("#featuresContainer input");
    let features = Array.from(checkboxes).map((checkbox) => checkbox.checked);

    eel.set_xlabels(features);
};
