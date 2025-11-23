

document.getElementById("registerForm").addEventListener("submit", async(eve) =>{

    eve.preventDefault();
    console.log(1);
    const formdata =  new FormData(document.getElementById("registerForm"));

    const ans = await fetch('/usernameexists',{
        
        method: 'POST',
        body: formdata
        
    });
    console.log(2);

    const exists = await ans.json();
    console.log(Boolean(exists.userexist));

    if(Boolean(exists.userexist) === true){
        console.log()
        alert("username is taken, Choose another");

    }else if(!Boolean(exists.userexist)){

        console.log(4);
        await fetch('/registeruser',{
            method: 'POST',
            body: formdata

        });

        window.location.href = '/';

    }


});

