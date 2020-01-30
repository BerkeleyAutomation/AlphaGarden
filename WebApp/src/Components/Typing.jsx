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
        const labelStyle = {
            position: 'absolute',
            width: 'max-content',
            fontFamily: 'Roboto Mono',
            textTransform: 'uppercase',
            letterSpacing: '6px',
            fontSize: '25px',
            fontWeight: 'regular',
            left: this.props.x + this.props.radius * 0.7,
            top: this.props.y + this.props.radius * 0.7,
            margin: 0
        }; 

        return (
            <p style={labelStyle}>{ this.state.text }</p>);
    }
  }

  export default Typing;