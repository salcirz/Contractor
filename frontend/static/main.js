console.log(1);
document.addEventListener("DOMContentLoaded", () => {  
     
    document.getElementById("file").addEventListener("change", (event) => {

        const file = event.target.files[0];
        const image = URL.createObjectURL(file);

        const center =  document.getElementById("main");
        center.style.backgroundImage = `url(${image})`;
        center.style.backgroundSize = "cover";
        center.style.backgroundPosition = "center";


    });


    document.getElementById("estimateform").addEventListener("submit",  async(ev) =>{

        ev.preventDefault();
        const form = new FormData(document.getElementById("estimateform"));
        const res = await fetch('/getprice', {

            method: 'POST',
            body: form

        });

        const data = await res.json();

        document.getElementById("joboutput").innerHTML = data.job;
   
        let count = 0;

        const interval = setInterval(() => {

            if(count <= data.price){
                document.getElementById("priceoutput").innerHTML = count;

            }   
            if(count <= data.time){
                document.getElementById("timeoutput").innerHTML = count;

            }  
            if(count <= data.cost){
                document.getElementById("costoutput").innerHTML = count;

            } 

            count++;

            if(count > data.price && count > data.time && count > data.cost){
                
                clearInterval(interval); 
            }

        },2);

        const images = await fetch('/getimageprice', {

            method: 'POST',
            body: form

        });

        const imagedata = await images.json();

    });


});