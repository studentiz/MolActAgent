import { staticFile } from "remotion";

export const Fonts: React.FC = () => {
  return (
    <style>
      {`
        @font-face {
          font-family: 'DM Sans';
          src: url('${staticFile("fonts/dm-sans.woff2")}') format('woff2');
          font-weight: 100 1000;
          font-style: normal;
          font-display: block;
        }
      `}
    </style>
  );
};
