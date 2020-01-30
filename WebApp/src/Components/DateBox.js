import React from 'react';
	
 const DateBox = ({x, y}) => {
    let today = new Date();

    const boxStyle = {
        position: 'absolute',
        width: 'max-content',
        textTransform: 'uppercase',
        letterSpacing: '3px',
        fontSize: '36px',
        fontWeight: 'lighter',
        top: y,
        left: x,
        border: '1px solid white',
        color: 'white',
        padding: '5px'
    }

    const dayStyle = {
        display: 'inline',
        borderRight: '1px solid white',
        padding: '5px',
    }

    const dateStyle = {
        display: 'inline',
        paddingLeft: 12
    }


    return(
        <div className="DateBox" style={boxStyle}>
            <div><p style={dayStyle}>Day</p><p style={dateStyle}>{today.getDate() + today.getMonth()}</p></div>
        </div>
    )    
}

export default DateBox;