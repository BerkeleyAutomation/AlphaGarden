import React from 'react';

 const DatePage = (props) => {

    if(props.nuc){setTimeout(props.endFunc, 2500)}
     
    const formatDate = (date) =>{
        var monthNames = [
          "January", "February", "March",
          "April", "May", "June", "July",
          "August", "September", "October",
          "November", "December"
        ];
        var d = date.getDate();
        var m = date.getMonth() + 1;
        var y = date.getFullYear();
        return monthNames[m - 1] + ' ' + (d <= 9 ? '0' + d : d) + ', ' + y ;
    }

    var currentDate = formatDate(new Date());

	 return(
        <div className="fade-in">
          <div className="IntroDate">
            <p id="jumbotron-date">January 01, 2020 - {currentDate}</p>
          </div>
        </div> 
		)    
}

export default DatePage;