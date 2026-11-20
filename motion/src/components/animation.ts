import { Easing, interpolate } from "remotion";

export const ease = Easing.bezier(0.22, 0.75, 0.2, 1);

export const reveal = (frame: number, start: number, duration = 18) =>
  interpolate(frame, [start, start + duration], [0, 1], {
    easing: ease,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

export const sceneOpacity = (frame: number, duration: number, fade = 12) =>
  Math.min(
    interpolate(frame, [0, fade], [0, 1], {
      easing: ease,
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
    interpolate(frame, [duration - fade, duration], [1, 0], {
      easing: ease,
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
  );

export const rise = (frame: number, start: number, distance = 22, duration = 20) => {
  const value = reveal(frame, start, duration);
  return {
    opacity: value,
    transform: `translateY(${(1 - value) * distance}px)`,
  };
};

export const draw = (frame: number, start: number, end: number) =>
  interpolate(frame, [start, end], [0, 1], {
    easing: ease,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
