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
        let y = this.props.y + this.props.radius * 0.7;
        if (y > this.props.startY + this.props.gridHeight) {
          y = this.props.y - this.props.radius * 0.7;
        }
        const labelStyle = {
            position: 'absolute',
            width: 'max-content',
            textTransform: 'uppercase',
            letterSpacing: '3px',
            fontSize: '30px',
            fontWeight: '200',
            left: this.props.x + this.props.radius * 0.7,
            top: y,
            margin: 0
        }; 

        return (
            <p style={labelStyle}>{ this.state.text }</p>);
    }
  }

  export default Typing;