import React from 'react';
import PageHeading from './PageHeading';
import Watercolor from './Watercolor';
import ReactMarkdown from 'react-markdown';

const About = () => {
  return (<div>
    <PageHeading title="Alphagarden" subtitle="About" />
    <Watercolor type="small" />
    <div className="about-content">
      <div className="compact-text">
        <p>Ken Goldberg and the AlphaGarden Collective (USA)</p>
        <p>Robot, garden, deep learning software.</p>
        <p>AlphaGarden.org</p>
      </div>
        
      <p>
        What is intelligence?  
      </p>

      <p>
        AlphaGarden is a robotic artwork that juxtaposes natural and artificial intelligence.  
        It invites viewers to reflect on the natural world and our role and place within it.  
      </p>

      <p>      
        Gardens strive to cultivate nourishment, sustenance, and in the best of cases, equilibrium with nature.  
        The Garden of Eden focused on  a tree of knowledge;  what is the status of knowledge today?   
        Can a robot learn to sustain a living garden? 
        AlphaGarden puts claims about the AI Revolution into the context of the Agricultural and Industrial Revolutions.   
        What should we make of a machine that uses AI to perform acts so deeply entwined with human history, mythology, and culture?
      </p>

      <p>
        AlphaGarden references AlphaGo and AlphaZero, the game-playing AI systems that have defeated world masters in recent years.  
        It builds on Goldberg’s widely known TeleGarden (1995-2004), where visitors interacted with a living garden via the Internet.  
        AlphaGarden places a robot into a living polyculture garden to study the nature of diversity and explore the limitations of artificial intelligence in the context of ecology and sustainability. 
      </p>

      <p>
        The project in progress consists of an automated robot that has been installed over a 3 meters long by 1.5 meters wide garden at the University of California at Berkeley. 
        Deep Learning AI policies attempt to control the three-axis robot that tends the garden, which includes edible plants and invasive species in a biodiverse polyculture environment. 
        The display will be updated with evolving images and data from the garden over the course of the exhibition.
      </p>

      <p>
        AlphaGarden does not assume the garden will flourish. 
        The robot will struggle.  
        It will over-water, under-prune; plants may die;  invasive species in the garden may take over.  
        The robot is being trained with simulations that run 100,000 times faster than nature.  
        But simulations only approximate reality.  
        There is enormous complexity in nature,  we haven’t been able to solve the three-body problem in thermodynamics and a garden has far more variables.  
        There will always be a distance between simulation and reality.  
        Evoking scenes from War Games and WALL-E, AlphaGarden exposes this “reality gap” with a growing garden and visually compelling display that evolves over time. 
      </p>

      <p>
        AlphaGarden foregrounds the beauty of nature and exposes the limitations of AI and robots.  
        Natural ecosystems of soils, microbes, seeds, insects, and plants precede humans and have their own forms of communication, behavior, knowledge, intelligence.  
        In a polyculture garden, plants both cooperate and compete for water, light, nutrients, and other resources. Some are invasive and, left alone, will outcompete other plants to transform a polyculture to a monoculture.  
        Even at a small scale, such an ecosystem is extraordinarily difficult to analyze or predict.  
      </p>

      <p>
        The project speaks to the environmental precipice we are on – as a culture and as a species. 
        Global temperatures are rising, we are seeing unprecedented natural disasters and apocalyptic fires. 
        Meanwhile, we are building machines that surpass human intelligence at specific tasks.  
        AlphaGarden is an evolving, suggestive meditation of what is current – and what lies ahead. 
      </p>

      <p>
        - Sarah Newman, Harvard metaLAB
      </p>
    </div>
  </div>)
}

export default About;