import React from 'react';
	
 const DateBox = (props) => {
    const dayStyle = {
        display: 'inline',
        borderRight: '1px solid white',
        padding: '5px',
        paddingLeft: '25px'
    }

    const dateStyle = {
        display: 'inline',
        paddingLeft: '25px',
        fontFamily: 'Roboto Thin',
    }

    var date1 = new Date("1/1/2020");
    var date2 = new Date();
    var timeDiff = Math.abs(date2.getTime() - date1.getTime());
    var diffDays = Math.ceil(timeDiff / (1000 * 3600 * 24)); 

    if (props.shouldDisplay === true) {
        return(
            <div className="date-box-row">
                <div className="date-box-row-item" id="date-box-div">
                    <div className="date-box"><p style={dayStyle}>Day</p><p style={dateStyle}>{diffDays}</p></div>
                </div>
                <div className="date-box-row-item">
                </div>
                <div className="date-box-row-item">
                </div>
            </div>
        )    
    } else {
        return null;
    }
}

export default DateBox;