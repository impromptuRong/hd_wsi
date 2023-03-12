document.getElementById("slide").addEventListener("change", loadslide);
function loadslide() {
    const form = document.getElementById("options");
    var url = form['slide'].value;
    console.log(url);
    window.location = url;
};

document.getElementById("model").addEventListener("change", loadmodel);
document.getElementById("device").addEventListener("change", loadmodel);
function loadmodel() {
    const form = document.getElementById("options");
    var url = (form.action + 
               '?' + 'model=' + form['model'].value +
               '&' + 'device=' + form['device'].value)
    console.log(url)
    fetch(url, {
        "method": "POST",
    }).then(resp => resp.json()).then(data => {
        form['model'].value = data['model'];
        form['device'].value = data['device'];
    });
};


document.getElementById("servicebtn").addEventListener("click", service);
async function service() {
    const btn = document.getElementById("servicebtn");
    if (btn.textContent === "Run") {
        var url = (btn.value + '?' + 'toggle=true');
        btn.textContent = "Stop";
    } else {
        var url = (btn.value + '?' + 'toggle=false');
    }
    console.log(url);
    await fetch(url, {
        "method": "POST",
    }).then(resp => resp.json()).then(data => {
        console.log(data);
        if (data['status'] != 'run') {
            btn.textContent = "Run";
        }
    });
};


document.getElementById("nucleislider").addEventListener("change", function () {
    const btn = document.getElementById("nucleislider");
    var overlayNuclei = btn.checked;
    var mpp = btn.value;

    const ele = OpenSeadragon.getElement('nucleibtn');
    ele.value = overlayNuclei;  // sync overlay

    displayNuclei(overlayNuclei, mpp, clearCache=true);
});


function displayNuclei(overlayNuclei, mpp=0.25, clearCache=false) {
    const viewer = OpenSeadragon.getViewer("view");

    var slideImage = viewer.world.getItemAt(0);
    var viewportzoom = viewer.viewport.getZoom();
    var imagezoom = slideImage.viewportToImageZoom(viewportzoom)
    var mppzoom = (mpp ? mpp : 0.25) / imagezoom

    var nucleiImage = viewer.world.getItemAt(1);
    if (nucleiImage) {
        if (mppzoom < 1.0 && overlayNuclei) {
            if (clearCache) {
                var source = nucleiImage.source;
                var bounds = nucleiImage.getBounds();
                viewer.addTiledImage({
                    tileSource: source,
                    index: 1,
                    replace: true,
                    x: bounds.x,
                    y: bounds.y,
                    width: bounds.width, 
                    opacity: 1.0,
                });
            }
            else {
                nucleiImage.setOpacity(1);
            }
        } else {
            nucleiImage.setOpacity(0);
        }
    }
};
