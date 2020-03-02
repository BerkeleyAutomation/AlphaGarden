import React from 'react';
	
 const DateBox = (props) => {
    const dayStyle = {
        borderRight: '1px solid white',
        padding: '5px',
        paddingLeft: '25px',
        fontSize: '30px',
        width: '50%',
        margin: 0
    }

    const dateStyle = {
        paddingLeft: '25px',
        fontFamily: 'Roboto Thin',
        fontSize: '30px',
        margin: 0,
        padding: '5px',
        paddingRight: '50px'
    }

    const dateBoxStyle = {
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '80%'
    }

    var date1 = new Date("1/1/2020");
    var date2 = new Date();
    var timeDiff = Math.abs(date2.getTime() - date1.getTime());
    var diffDays = Math.ceil(timeDiff / (1000 * 3600 * 24)); 

    if (props.shouldDisplay === true) {
        return(
            <div className="date-box-row">
                <div className="date-box-row-item" id="date-box-div">
                    <div className="date-box" style={dateBoxStyle}>
                        <p style={dayStyle}>Day</p>
                        <p style={dateStyle}>{diffDays}</p>
                    </div>
                </div>
                <div className="date-box-row-item">
                    <p>Click to zoom in on any section</p>
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