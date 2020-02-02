import React from 'react';
	
 const DateBox = ({x, y}) => {
    let today = new Date();

    const boxStyle = {
        position: 'absolute',
        width: 'max-content',
        textTransform: 'uppercase',
        letterSpacing: '3px',
        font: 'Roboto Mono',
        fontSize: '36px',
        fontWeight: 'regular',
        top: y,
        left: x,
        border: '2px solid white',
        color: 'white',
        padding: '5px'
    }

    const dayStyle = {
        display: 'inline',
        borderRight: '2px solid white',
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