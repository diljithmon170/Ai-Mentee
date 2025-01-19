import React, { useState } from "react";
import styled from "styled-components";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";
import { Pagination, Navigation } from "swiper/modules";
import FlashCardItem from "./FlashcardItem";

const StyledSwiper = styled.div`
  .swiper {
    width: 100%;
    height: 50vh;
  }

  .swiper-slide {
    text-align: center;
    font-size: 18px;
    background: #ededed;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .swiper-slide img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .swiper {
    margin-left: auto;
    margin-right: auto;
  }
`;

export default function App() {
  const [isFlipped, setIsFlipped] = useState(false);
  const handleClick = () => {
    setIsFlipped(!isFlipped);
  };
  return (
    <StyledSwiper>
      
      <Swiper
        slidesPerView={1}
        spaceBetween={30}
        loop={true}
        pagination={{ clickable: true }}
        navigation={true}
        modules={[Pagination, Navigation]}
        className="mySwiper"
      >
        
        <SwiperSlide>
          <FlashCardItem isFlipped={isFlipped} handleClick={handleClick} />
        </SwiperSlide>
        <SwiperSlide>Slide 2</SwiperSlide> <SwiperSlide>Slide 3</SwiperSlide>
        <SwiperSlide>Slide 4</SwiperSlide> <SwiperSlide>Slide 5</SwiperSlide>
        <SwiperSlide>Slide 6</SwiperSlide> <SwiperSlide>Slide 7</SwiperSlide>
        <SwiperSlide>Slide 8</SwiperSlide> <SwiperSlide>Slide 9</SwiperSlide>
      </Swiper>
    </StyledSwiper>
  );
}