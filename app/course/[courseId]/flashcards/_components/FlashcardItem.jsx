import React from 'react'
import ReactCardFlip from 'react-card-flip';

function FlashCardItem({isFlipped, handleClick, flashCard}) {
  return (
    <div className="pb-28 h-[50vh]">
      <ReactCardFlip isFlipped={isFlipped} flipDirection="vertical">
        <div className="p-7 bg-primary text-white flex shadow-lg items-center justify-center rounded-lg cursor-pointer h-[250px] w-[250px] md:h-[400px] md:w-[350px]" onClick={handleClick}>
          {flashCard.front}
        </div>
        <div className="p-7 text-primary bg-white shadow-lg flex items-center justify-center rounded-lg cursor-pointer h-[250px] w-[250px] md:h-[400px] md:w-[350px]" onClick={handleClick}>
          {flashCard.back}
        </div>
      </ReactCardFlip>
    </div>
  );
}

export default FlashCardItem