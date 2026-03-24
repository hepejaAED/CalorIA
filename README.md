<table border="0">
  <tr>
    <td>
      <img src="icon.png" alt="CalorIA" width="200">
    </td>
    <td>
      <h1> Pipeline Nutricional Inteligente</h1>
    </td>
  </tr>
</table>

---

# Ejemplo

Aqui abría que añadir el input y el output, poner imágenes

# Objetivo

El objetivo de este proyecto es desarrollar un sistema basado en Deep Learning capaz de estimar las calorías y macronutrientes de una comida a partir de una imagen, integrando de forma coherente información visual y textual proporcionada por el usuario. Para ello, se plantea el diseño de un pipeline multimodal que permita identificar los ingredientes presentes en el plato, inferir su contexto culinario y estimar cantidades realistas, combinando el uso de modelos de visión-lenguaje con técnicas de procesamiento de texto. Asimismo, el sistema busca apoyarse en bases de datos nutricionales y mecanismos de estimación cuando sea necesario, con el fin de obtener una aproximación fiable de los valores nutricionales, manteniendo una arquitectura modular que facilite su mejora y escalabilidad.

# Arquitectura del Pipeline

<p align="center">
  <img src="pipeline_arquitectura.excalidraw.svg" alt="Pipeline de IA Nutricional" width="90%">
</p>

*Diagrama detallado del flujo de datos desde la entrada multimodal hasta el resultado final.*


# Tecnologías utilizadas



# Componentes del sistema

Describir las funciones y las clases

# Metodología

- **Input**: Imagen de la comida y un texto opcional del usuario por si es necesario completar la información.
- el texto se preprocesa con un Text to text utilizando Qwen2.5-3B-Instruct con el objetivo de traducir el texto del usuario 

# Uso

<p align="center">
  <img src="imagenes/foto.jpg" alt="" width="50%">
</p>



# Limitaciones

La estimación de cantidades a partir de la imagen es una estimación y depende de la calidad y la perspectiva de la imagen. Además, la detección de alimentos ocultos o poco visibles puede resultar imprecisa.
La cobertura de las bases de datos nutricionales no siempre es completa, requiriendo estimaciones adicionales. Por último, el coste computacional del sistema es elevado, lo que se traduce en tiempos de inferencia altos y limita su aplicabilidad en entornos en tiempo real.


# Futuros trabajos



# Bibliografía

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={[https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)}, 
}

@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}

```
