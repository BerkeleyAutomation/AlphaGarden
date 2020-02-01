import React from 'react';

 const DatePage = (props) => {

    if(props.nuc){setTimeout(props.endFunc, 5000)}
     
    const formatDate = (date) =>{
        var monthNames = [
          "Jan", "Feb", "Mar",
          "Apr", "May", "June", "July",
          "Aug", "Sept", "Oct",
          "Nov", "Dec"
        ];
        var d = date.getDate();
        var m = date.getMonth() + 1;
        var y = date.getFullYear();
        return monthNames[m - 1] + ' ' + (d <= 9 ? '0' + d : d) + ' ' + y ;
    }

    var currentDate = formatDate(new Date());

	 return(
        <div className="fade-in">
          <div className="IntroDate">
            <p id="jumbotron-date">Jan 01 2020 - {currentDate}</p>
          </div>
        </div> 
		)    
}

export default DatePage;