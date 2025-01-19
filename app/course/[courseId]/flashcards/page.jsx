"use client";

import axios from "axios";
import { useParams } from "next/navigation";
import React, { useEffect, useState } from "react";
import styled from "styled-components";
import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "swiper/css/pagination";
import "swiper/css/navigation";
import { Pagination, Navigation } from "swiper/modules";
import FlashcardItem from "./_components/FlashcardItem";

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

function Flashcards() {
  const { courseId } = useParams();
  const [flashCards, setFlashCards] = useState([]);
  const [flippedStates, setFlippedStates] = useState({});

  useEffect(() => {
    GetFlashCards();
  }, []);

  const GetFlashCards = async () => {
    const result = await axios.post("/api/study-type", {
      courseId: courseId,
      studyType: "Flashcard",
    });
    setFlashCards(result.data);
    console.log("Flashcard", result.data);
  };

  const handleClick = (index) => {
    setFlippedStates((prevState) => ({
      ...prevState,
      [index]: !prevState[index],
    }));
  };

  return (
    <div className="h-screen">
      <h2 className="font-bold text-2xl">Flashcards</h2>
      <p>Help you to remember your</p>
      <div className="mt-28">
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
            {flashCards.content?.map((flashCard, index) => (
              <SwiperSlide key={index}>
                <FlashcardItem
                  isFlipped={flippedStates[index]}
                  handleClick={() => handleClick(index)}
                  flashCard={flashCard}
                />
              </SwiperSlide>
            ))}
          </Swiper>
        </StyledSwiper>
      </div>
    </div>
  );
}

export default Flashcards;
