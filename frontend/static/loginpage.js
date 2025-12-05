//js file to login user

document.getElementById("loginForm").addEventListener("submit", async (ev)=>{

    ev.preventDefault();

    form = new FormData(document.getElementById("loginForm"));

    const correctlogin = await fetch('/loginuser',{

        method: 'POST',
        body: form,

    });

    const data = await correctlogin.json();

    if(data.correct){
        window.location.href = '/';
    }else{
        alert("username or password is incorrect");
    }
    



});