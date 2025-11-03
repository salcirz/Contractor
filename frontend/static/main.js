console.log(1);
document.getElementById("file").addEventListener("change", (event) => {

    const file = event.target.files[0];
    const image = URL.createObjectURL(file);

    const center =  document.getElementById("main");
    center.style.backgroundImage = `url(${image})`;
    center.style.backgroundSize = "cover";
    center.style.backgroundPosition = "center";


});

