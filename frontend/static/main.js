console.log(1);



document.addEventListener("DOMContentLoaded", async(events) => {  

    let signedin = false;
    const perfEntries = performance.getEntriesByType("navigation");
    if (perfEntries.length > 0 && perfEntries[0].type === "reload") {
        fetch("/clearsession");
        console.log("Page was refreshed!");
    }
    
    events.preventDefault();
    let sessiondata = await fetch('/getsessioninfo');
    let sessiondatajson = await sessiondata.json();

    if(sessiondatajson && sessiondatajson.username !== undefined){

        signedin = true; 
        console.log("one");
        document.getElementById("loginlogo").style.opacity = "1";
        document.getElementById("loginlogo").style.pointerEvents = "all";
        document.getElementById("initial").innerHTML = sessiondatajson.username.toUpperCase()[0];
        document.getElementById("person").style.opacity = "0";
        console.log(sessiondatajson.username);
        document.getElementById("person").style.opacity = "none";
        
    }else{
         document.getElementById("person").style.opacity = "1";
        
    }
    

    document.getElementById("file").addEventListener("change", (event) => {


        const file = event.target.files[0];
        const image = URL.createObjectURL(file);

        const center =  document.getElementById("main");
        center.style.backgroundImage = `url(${image})`;
        center.style.backgroundSize = "cover";
        center.style.backgroundPosition = "center";

        document.getElementById("cloudlabel").style.backgroundColor = "transparent";
        document.getElementById("cloudlabel").style.opacity = 0;
        document.getElementById("cloudlogo").style.backgroundColor = "transparent";

    });



    document.getElementById("estimateform").addEventListener("submit",  async(ev) =>{

        ev.preventDefault();
        const form = new FormData(document.getElementById("estimateform"));
        const res = await fetch('/getprice', {

            method: 'POST',
            body: form

        });

        const data = await res.json();
   
        let count = 0;

        document.getElementById("timeoutput").innerHTML = data.time;
        const interval = setInterval(() => {

            if(count <= data.price){
                document.getElementById("priceoutput").innerHTML = count + '$';

            }   
            
            if(count <= data.cost){
                document.getElementById("costoutput").innerHTML = count + '$';

            } 

            count++;

            if(count > data.price && count > data.time && count > data.cost){
                
                clearInterval(interval); 
            }

        },2);
        
        const img = await fetch('/getimageprice', {

            method: 'POST',
            body: form

        });

        document.getElementById("reportbutton").style.opacity = '100%';
        document.getElementById("reportbutton").style.pointerEvents = 'all';
        document.getElementById("reportbutton").style.pointerEvents = 'all';

        document.getElementById('reportbutton').addEventListener("click", () =>{
            window.location.href = '/reportpage';
       
        });

        
    });
    
    document.getElementById('numberinput').addEventListener('input', function(e) {
        this.value = this.value.replace(/[^0-9.]/g, ''); 
    });

    const popup = document.getElementById("popup");
    const links = document.getElementsByClassName("poplink");

    document.getElementById("personbut").addEventListener("click", async(eventt)=>{

        if(!signedin) window.location.href = '/registerpage.html';
        else popup.style.opacity = 1;

        

        for (let i = 0; i < links.length; i++) {
            links[i].style.pointerEvents = `all`;
        }

    });

    popup.addEventListener("mouseleave",()=>{
        popup.style.opacity = 0;
        for (let i = 0; i < links.length; i++) {
            links[i].style.pointerEvents = `none`;
        }

    });

    

  

});