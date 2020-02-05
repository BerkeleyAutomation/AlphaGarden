import React from 'react';

class Typing extends React.Component {

    static defaultProps = {
      dataText: []
    }
  
    constructor(props) {
      super(props);
  
      this.state = {
        text: '',
        typingSpeed: 30
      }
    }
  
    componentDidMount() {
      this.handleType();
    }
  
    handleType = () => {
      const { dataText } = this.props;
      const { text, typingSpeed } = this.state;
  
      this.setState({
        text: dataText[0].substring(0, text.length + 1),
        typingSpeed: 80 
      });
  
      setTimeout(this.handleType, typingSpeed);
    };
  
    render() {   
        let y = this.props.y + this.props.radius * 0.4;
        if (y > this.props.startY + this.props.gridHeight) {
          y = this.props.y - this.props.radius * 0.4;
        }
        const labelStyle = {
            position: 'absolute',
            width: 'max-content',
            fontFamily: 'Roboto Thin',
            textTransform: 'uppercase',
            fontSize: '30px',
            letterSpacing: '10px',
            left: this.props.x + this.props.radius,
            top: y,
            margin: 0
        }; 

        return (
            <p style={labelStyle}>{ this.state.text }</p>);
    }
  }

  export default Typing;