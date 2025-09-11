create database data ;

use data;
create table pictures(

    id INT auto_increment PRIMARY KEY,
    price double, 
    typ varchar(25),
    path varchar(255)


);