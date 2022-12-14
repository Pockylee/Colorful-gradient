

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Pockylee/Colorful-gradient">
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

<h3 align="center">An Microscope Image Auto-Focus Method based on Colorful-Gradient</h3>

  <p align="center">
    This is a project mainly focus on the issue when automated microscope dealing with blood samnple images.
    We proposed a brand new and effective algorithm as for automated microscope focus method solution.
    <br />
    <br />
    <a href="https://github.com/Pockylee/Colorful-gradient">View Demo</a>
    ·
    <a href="https://github.com/Pockylee/Colorful-gradient/issues">Report Bug</a>
    ·
    <a href="https://github.com/Pockylee/Colorful-gradient/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This is a project mainly focus on the issue when automated microscope dealing with blood sample images.
There are three components in this algorithm which are: Sharpness value, Colorfulness value, and Color cast score.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Installation

Clone the repo
   ```sh
   git clone https://github.com/Pockylee/Colorful-gradient.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

While observing a blood sample under the micoscope, the conventional sharpness operator will be effected serously by the 

_For more examples, please refer to the [Documentation](https://example.com)_

In this version, we have only one program to calculate the focus valuem, which is colorful_gradient.py
<br/>


To check every argument in this program which can be modify:
   ```sh
    python3 colorful_gradient.py --help
   ```
You'll get the output of the arguments information:

   ```sh
    -t CASE_TYPE, --case_type CASE_TYPE
                        Type of cases in the input image directory.(single/multiple)
    -i DIR_PATH, --dir_path DIR_PATH
                        Please enter your image directory path here.
    -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Please enter the output path where you want to store your explicit image in
                        each cases
    -p, --parallel_compute
                        Choose to calculate the colorful-gradient in parallel computing mode or
                        not.(True/False)
   ```
Due to the privacy policy, we are not able to upload our experiment data images to github. Please prepare your own dataset to run this program!

After start running the program, the program will show the progress to make sure that it is not crash.
Including showing the Input Path, Explicit Index, and the Output Path.
   ```sh
  Program start
  Task Type:  Multiple
  Parallel Compute:  True
  ===================================================
  Input Path:  ./images/multiple/case2
  Explicit Index:  0
  Output Path:  ./output/case2/P_05_220119_161941.jpg
  ===================================================
  Input Path:  ./images/multiple/case_1
  Explicit Index:  0
  Output Path:  ./output/case_1/P_21_220119_162014.jpg
  ===================================================
  End of the program. See you next time!
   ```
The program can be executed on both macOS and Linux.

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Brian Li - brian7149@gmail.com

Project Link: [https://github.com/Pockylee/Colorful-gradient](https://github.com/Pockylee/Colorful-gradient)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Pockylee/Colorful-gradient.svg?style=for-the-badge
[contributors-url]: https://github.com/Pockylee/Colorful-gradient/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Pockylee/Colorful-gradient.svg?style=for-the-badge
[forks-url]: https://github.com/Pockylee/Colorful-gradient/network/members
[stars-shield]: https://img.shields.io/github/stars/Pockylee/Colorful-gradient.svg?style=for-the-badge
[stars-url]: https://github.com/Pockylee/Colorful-gradient/stargazers
[issues-shield]: https://img.shields.io/github/issues/Pockylee/Colorful-gradient.svg?style=for-the-badge
[issues-url]: https://github.com/Pockylee/Colorful-gradient/issues
[license-shield]: https://img.shields.io/github/license/Pockylee/Colorful-gradient.svg?style=for-the-badge&kill_cache=1
[license-url]: https://github.com/Pockylee/Colorful-gradient/blob/master/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/po-yi-brian-li-44bbab18a
[product-screenshot]: readme_image/parallel_workflow.jpg

[Python-url]: https://www.python.org/
[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
