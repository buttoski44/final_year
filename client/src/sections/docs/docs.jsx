import React, { useState } from "react";
import Block from "./block";
import Indices from "./indices";
import { twMerge } from "tailwind-merge";
import { Button } from "@/components/ui/button";
import { ChevronsLeft } from "lucide-react";

// import components
import Heading from "./heading";
import SubHeading from "./subheading";
import Text from "./text";
import CodeBlock from "./codeblock";
import CodeWord from "./codeword";
import Underline from "./underline";
import List from "./list";
import Image from "./image";
import Callout from "./callout";
import Architecture from "@/assets/architecture";
import Client from "@/assets/client";

// import codes
import { code2 } from "@/codes/server_modules";
import { code3 } from "@/codes/express_app";
import { code4 } from "@/codes/api";
import { model1 } from "@/codes/model1";
import { model2 } from "@/codes/traning";
import { model3 } from "@/codes/model3";
import { model4 } from "@/codes/model4";
import { model5 } from "@/codes/model5";
import { model6 } from "@/codes/model6";
import { model7 } from "@/codes/model7";
import { model8 } from "@/codes/model8";

const docBlocks = [
  {
    title: "Introduction",
    code: (
      <Block key="inro">
        <Heading>Introduction</Heading>
        <SubHeading id="background">1.Background</SubHeading>
        <Text>
          Welcome to the future of driving! Picture this: you're cruising down
          the highway, wind in your hair, tunes blasting, and suddenly your car
          senses your every move, predicting your next action before you even
          make it. How? With our revolutionary Cognitive Driver Action
          Recognition System!
        </Text>
        <Text>
          But hold on, what's that, you ask? Well, imagine your car isn't just a
          machine, but a mind-reader, understanding your intentions and
          reactions better than your best friend. This isn't some sci-fi
          fantasy; it's the power of artificial intelligence in action.
        </Text>
        <Text>
          Now, buckle up and join us on this thrilling journey. Our project
          isn't just about pushing boundaries; it's about building a community
          of curious minds eager to dive into the world of AI. Together, we'll
          unravel the mysteries of cognitive driver action recognition, learn,
          teach, and most importantly, have fun while doing it.
        </Text>
        <Text>
          So, gear up, fellow explorers! Let's drive into the future, one
          curious mind at a time. Welcome aboard! 🚗💨 #AIOnTheRoad
        </Text>
        <SubHeading id="scope">2.Scope</SubHeading>
        <List className="list-decimal">
          <li>
            <CodeWord classNames="text-teal-700">
              Algorithm Development:
            </CodeWord>{" "}
            Focus on developing machine learning algorithms capable of
            recognizing and predicting driver actions based on input data from
            sensors such as cameras, lidar, radar, and other onboard sensors.
            Explore various techniques such as deep learning, computer vision,
            and sensor fusion.
          </li>
          <li>
            <CodeWord classNames="text-violet-700">
              Data Collection and Annotation:
            </CodeWord>{" "}
            Gather a diverse dataset of driving scenarios, including various
            road conditions, weather conditions, and driver behaviors. Annotate
            the dataset with labels indicating different driver actions.
            Encourage contributions from the community to expand and diversify
            the dataset.
          </li>
          <li>
            <CodeWord classNames="text-rose-700">
              Model Training and Optimization:
            </CodeWord>{" "}
            Train the developed algorithms using the annotated dataset to
            optimize their performance in accurately recognizing driver actions
            in real-time. Experiment with different architectures,
            hyperparameters, and training strategies to improve model accuracy
            and efficiency.
          </li>
          <li>
            <CodeWord classNames="text-emerald-700">
              Integration with Open Source Frameworks:
            </CodeWord>{" "}
            Integrate the developed models into open-source frameworks and
            libraries for ease of use and accessibility. Ensure compatibility
            with popular programming languages and platforms to encourage
            widespread adoption and contributions from the community.
          </li>
          <li>
            <CodeWord classNames="text-amber-700">
              Documentation and Tutorials:
            </CodeWord>{" "}
            Provide comprehensive documentation, tutorials, and example code to
            guide users through the process of using and contributing to the
            project. Cover topics such as data collection, model training,
            evaluation, and deployment, catering to users with varying levels of
            expertise.
          </li>
          <li>
            <CodeWord classNames="text-sky-300">Community Engagement:</CodeWord>{" "}
            Foster an open-source community around the project by actively
            engaging with contributors, answering questions, and reviewing
            contributions. Organize virtual meetups, hackathons, and
            collaborative projects to facilitate learning and collaboration
            among community members.
          </li>
          <li>
            <CodeWord classNames="text-red-700">Educational Outreach:</CodeWord>{" "}
            Promote the project through educational channels such as online
            courses, workshops, and university collaborations. Provide
            opportunities for students and enthusiasts to get involved in
            hands-on learning experiences and real-world projects related to
            artificial intelligence and computer vision.
          </li>
          <li>
            <CodeWord classNames="text-lime-700">
              Code Quality and Maintenance:
            </CodeWord>{" "}
            Maintain high standards of code quality, documentation, and testing
            to ensure the reliability and scalability of the project. Implement
            continuous integration and automated testing pipelines to streamline
            the development process and prevent regressions.
          </li>
          <li>
            <CodeWord classNames="text-sky-700">
              License and Governance:
            </CodeWord>{" "}
            Choose an appropriate open-source license for the project to ensure
            that it remains freely available for anyone to use, modify, and
            distribute. Establish transparent governance policies and
            decision-making processes to promote inclusivity and
            community-driven development.
          </li>
        </List>
        <Text>
          By focusing on these aspects within the project scope, we aim to
          develop an open-source Cognitive Driver Action Recognition System
          while fostering a collaborative learning environment and empowering
          individuals to contribute to the advancement of artificial
          intelligence and its applications in the automotive industry.
        </Text>
        <SubHeading id="architecture">3.Architecture</SubHeading>
        <Image>
          <Architecture></Architecture>
        </Image>
      </Block>
    ),
  },
  {
    title: "Server",
    code: (
      <Block key="server">
        <Heading>Server</Heading>
        <SubHeading id="modules">1. Importing Required Modules:</SubHeading>
        <Text>
          The Client-Server Model, a linchpin in modern computing paradigms,
          offers a structured framework for distributing tasks and resources
          across interconnected devices. Within the context of our cognitive
          driver action recognition system, this model serves as the cornerstone
          of our architectural design, facilitating efficient communication and
          collaboration between disparate components. Through this introduction,
          we embark on a journey to unravel the intricacies of our system's
          architecture, shedding light on its underlying mechanisms and the
          pivotal role played by the client-server model.
        </Text>
        <Text>
          Within this exposition, we delve into the anatomy of our system's code
          architecture, meticulously engineered to encapsulate the complexities
          of cognitive computing while ensuring scalability and reliability. By
          adopting the client-server model, we achieve a symbiotic relationship
          between client endpoints, such as onboard sensors and computing
          devices, and server nodes housing the analytical prowess and data
          repositories. This symbiosis fosters a dynamic ecosystem wherein
          real-time sensor data is seamlessly transmitted, processed, and
          analyzed to infer actionable insights regarding driver behavior and
          environmental dynamics.
        </Text>
        <Underline>
          Sure, let's break down the provided code step by step:
        </Underline>
        <CodeBlock file={code2} />
        <Underline>Here, the code imports necessary modules:</Underline>
        <List>
          <li>
            <CodeWord>express:</CodeWord> for creating the web server.
          </li>
          <li>
            <CodeWord classNames="text-green-700">multer:</CodeWord> for
            handling multipart/form-data, primarily used for uploading files.
          </li>
          <li>
            <CodeWord classNames="text-blue-700">
              @tensorflow/tfjs-node:
            </CodeWord>{" "}
            TensorFlow.js for Node.js, allowing the usage of TensorFlow models
            in a Node.js environment.
          </li>
          <li>
            <CodeWord classNames="text-yellow-700">cors:</CodeWord> for enabling
            CORS (Cross-Origin Resource Sharing) middleware, which allows
            restricting resources on a web page to be requested from another
            domain outside the domain from which the first resource was served.
          </li>
        </List>
        <SubHeading id="express">
          2. Creating an Express App , Loading Pre-Trained Model & Configuring
          Multer:
        </SubHeading>
        <CodeBlock file={code3}></CodeBlock>
        <Text>
          This section loads a pre-trained model using TensorFlow.js's
          <CodeWord bg={true}>loadLayersModel</CodeWord>
          function. It's wrapped inside an asynchronous function
          <CodeWord bg={true}>(async () =&gt; {})</CodeWord>
          to use <CodeWord bg={true}>await</CodeWord> for loading the model
          asynchronously. The path to the model JSON file is specified in{" "}
          <CodeWord bg={true}>modelPath</CodeWord>.
        </Text>
        <Text>
          Multer is configured to use in-memory storage (multer.memoryStorage())
          for handling file uploads. upload is an instance of Multer middleware
          configured with the storage settings.
        </Text>
        <SubHeading id="api">
          3. Handling POST Requests to '/classify' Endpoint:
        </SubHeading>
        <CodeBlock file={code4}></CodeBlock>
        <Text>
          This section defines a route{" "}
          <CodeWord bg={true}>(/classify)</CodeWord> to handle POST requests.
          The endpoint expects a single file upload named{" "}
          <CodeWord bg={true}>'image'</CodeWord>.
        </Text>
        <Text>
          Inside the request handler, it checks if a file was uploaded{" "}
          <CodeWord bg={true}>(if (!req.file))</CodeWord>. If no file is
          uploaded, it returns a 400 status code with an error message.
        </Text>
        <Text>
          If a file is uploaded, it decodes the image from base64 and resizes it
          to the required dimensions. Then, it expands the dimensions to match
          the input shape required by the model. After preprocessing, it makes
          predictions using the loaded model and sends the predicted class in
          the response.
        </Text>
      </Block>
    ),
  },
  {
    title: "Client",
    code: (
      <Block>
        <Heading>Client</Heading>
        <SubHeading id="much">Not So Much</SubHeading>
        <Image>
          <Client></Client>
        </Image>
      </Block>
    ),
  },
  {
    title: "Model",
    code: (
      <Block>
        <Heading>Model</Heading>
        <SubHeading id="traditional">1. Traditional CNN</SubHeading>
        <Text>
          Traditional Convolutional Neural Networks (CNNs), while highly
          effective for certain tasks, have some limitations that restrict their
          use in certain scenarios:
        </Text>
        <List>
          <li>
            <CodeWord classNames="text-yellow-700">Fixed Input Size:</CodeWord>{" "}
            Traditional CNNs typically require fixed-size input images. This
            means they struggle with varying input sizes, making them less
            flexible for tasks where input dimensions may vary widely (e.g.,
            object detection in images of different resolutions).
          </li>
          <li>
            <CodeWord classNames="text-blue-900">
              Lack of Spatial Attention:
            </CodeWord>{" "}
            Traditional CNNs process entire images with equal importance. They
            lack the ability to focus on specific regions of interest within an
            image, which can be crucial for tasks like fine-grained object
            recognition or scene understanding.
          </li>
          <li>
            <CodeWord classNames="text-green-700">
              Limited Contextual Understanding:
            </CodeWord>{" "}
            CNNs process local features effectively but may struggle with
            understanding global context in complex scenes. This limitation can
            hinder their performance in tasks requiring holistic understanding,
            such as scene understanding or image captioning.
          </li>
          <li>
            <CodeWord classNames="text-purple-700">Overfitting:</CodeWord> CNNs
            often have a large number of parameters, making them prone to
            overfitting, especially when dealing with limited training data.
            Regularization techniques such as dropout and weight decay are
            commonly used to mitigate this issue, but they may not always be
            sufficient.
          </li>
          <li>
            <CodeWord classNames="text-voilet-700">
              Computationally Intensive:
            </CodeWord>{" "}
            Traditional CNNs can be computationally intensive, especially when
            dealing with deep architectures and large datasets. Training and
            deploying these models may require significant computational
            resources, limiting their accessibility in resource-constrained
            environments.
          </li>
          <li>
            <CodeWord classNames="text-sky-700">
              Limited Robustness to Variations:
            </CodeWord>{" "}
            CNNs may struggle with variations in input data, such as changes in
            lighting conditions, viewpoint changes, occlusions, and background
            clutter. While data augmentation techniques can help address some of
            these challenges, they may not fully generalize across diverse
            conditions.
          </li>
          <li>
            <CodeWord classNames="text-pink-700">Semantic Gap:</CodeWord> CNNs
            may not capture high-level semantic concepts effectively. While they
            excel at learning low-level features, extracting meaningful semantic
            representations requires additional model architectures or
            post-processing steps.
          </li>
        </List>
        <Text>
          Despite these limitations, traditional CNNs have demonstrated
          remarkable performance in various computer vision tasks and remain a
          cornerstone of deep learning research and applications. Many of these
          limitations are actively being addressed through advancements in model
          architectures, training techniques, and the integration of additional
          components like attention mechanisms and graph structures.
        </Text>
        <SubHeading id="creation">2. Model Creation</SubHeading>
        <Text>
          A sequential model is created using
          <CodeWord classNames="text-green-800" bg={true}>
            models.Sequential()
          </CodeWord>
          . This is a common way to define models in Keras where layers are
          added sequentially.
        </Text>
        <Underline>Convolutional Layers (CNN Part):</Underline>
        <Text>
          The core of the model involves several convolutional layers:
        </Text>
        <List>
          <li>
            <CodeWord>Conv2D:</CodeWord> This defines a 2D convolutional layer
            with -{" "}
            <CodeWord classNames="text-gray-300/50" bg={true}>
              32
            </CodeWord>{" "}
            filters: This determines the number of learnable features the layer
            can detect in the input image. - Kernel size of{" "}
            <CodeWord classNames="text-gray-300/50" bg={true}>
              (3, 3):
            </CodeWord>
            This defines the size of the filter that slides across the image to
            extract features. - Activation of{" "}
            <CodeWord classNames="text-gray-300/50" bg={true}>
              'relu'
            </CodeWord>{" "}
            : This applies the Rectified Linear Unit (ReLU) activation function,
            which promotes non-linearity. -{" "}
            <CodeWord classNames="text-gray-300/50" bg={true}>
              input_shape=(240,240,3)
            </CodeWord>
            (assuming this is defined elsewhere): This specifies that the model
            expects input images to be 240x240 pixels with 3 color channels
            (RGB).
          </li>
          <li className="pb-10">
            <CodeWord>BatchNormalization:</CodeWord> This technique helps
            normalize activations of the previous layer, improving training
            stability.
          </li>
          <Callout className="-translate-x-12">
            Conv2D & BatchNormalization are repeated within each CNN block,
            potentially extracting different levels of features.
          </Callout>
          <li className="pt-10">
            <CodeWord>MaxPooling2D:</CodeWord> This layer downsamples the
            feature maps by taking the maximum value within a specified window
            (here, pool_size of (2,2)) which reduces the dimensionality of data
            and helps control overfitting.
          </li>
          <li>
            <CodeWord>Dropout:</CodeWord> This layer randomly drops a certain
            percentage of activations (here, 0.3 or 0.5 depending on the layer)
            during training to prevent overfitting.
          </li>
        </List>
        <Text>Dense Layers & Output:</Text>
        <List>
          <li>
            <CodeWord>Flatten:</CodeWord> This layer reshapes the data from a
            multi-dimensional tensor into a single feature vector.
          </li>
          <li>
            <CodeWord>Dense:</CodeWord> This defines a fully-connected layer
            with:
            <List className="pt-6">
              <li>
                <CodeWord bg={true}>units=512</CodeWord> or{" "}
                <CodeWord bg={true}>128</CodeWord>: This specifies the number of
                neurons in the layer. These layers learn more complex
                relationships between features.
              </li>
              <li>
                Activation of <CodeWord bg={true}>'relu'</CodeWord>: Again, the
                ReLU activation is used.
              </li>
            </List>
          </li>
          <li>
            Another dropout layer (0.25) is added for further regularization.
          </li>
          <li>
            Finally, a dense layer with<CodeWord bg={true}>10 units</CodeWord>{" "}
            and <CodeWord bg={true}>'softmax'</CodeWord> activation is used for
            the output. Since there are 10 units, the model likely predicts
            probabilities for 10 different image classes (assuming this is
            defined elsewhere).Softmax activation ensures the output
            probabilities sum to 1.
          </li>
        </List>
        <CodeBlock file={model1}></CodeBlock>
        <SubHeading id="training">3. Training</SubHeading>
        <List>
          <li>
            <CodeWord classNames="text-rose-800">model.compile():</CodeWord>
             This configures the model for training by specifying:
            <List className="pt-10">
              <li>
                <CodeWord classNames="bg-transperant font-bold">
                  Loss function:
                </CodeWord>
                 <CodeWord bg={true}>'categorical_crossentropy'</CodeWord>,
                which is commonly used for multi-class classification problems.
              </li>
              <li>
                <CodeWord classNames="bg-transperant font-bold">
                  Metrics:
                </CodeWord>
                 <CodeWord bg={true}>['accuracy']</CodeWord>
                 to track accuracy during training.
              </li>
              <li>
                <CodeWord classNames="bg-transperant font-bold">
                  Optimizer:
                </CodeWord>
                 <CodeWord bg={true}>'adam'</CodeWord>
                 (Adaptive Moment Estimation), a popular optimizer for gradient
                descent.
              </li>
            </List>
          </li>
          <li>
            <CodeWord classNames="text-orange-300">EarlyStopping:</CodeWord> A
            callback function is defined using callbacks.EarlyStopping to
            monitor validation accuracy. If validation accuracy doesn't improve
            for 5 epochs (patience), training stops to prevent overfitting.
          </li>
          <li>
            <CodeWord classNames="text-purple-500">
              Batch Size & Epochs:
            </CodeWord>
             The model is trained with a batch_size of 40 images and
            for n_epochs of 10 iterations over the entire training data.
          </li>
          <li>
            <CodeWord classNames="text-stone-500">model.fit():</CodeWord> This
            function performs the actual training using the defined
            configuration, training data (x_train, Y_train), validation data
            (x_test, Y_test), batch size, epochs, verbosity (showing training
            progress), and the early stopping callback.
          </li>
        </List>
        <CodeBlock file={model2}></CodeBlock>
        <SubHeading id="transfer">
          4. Image Augmentation for Transfer Learning Models
        </SubHeading>
        <Text>
          This code snippet is primarily focused on preparing data
          transformations for training, validation, and testing phases in a
          computer vision task using PyTorch's torchvision module.
        </Text>
        <Underline>Here's a breakdown of what the code does:</Underline>
        <List>
          <li>
            <CodeWord>Import Libraries:</CodeWord> The necessary libraries are
            imported, presumably already imported somewhere else in the
            codebase. These include torchvision.datasets.ImageFolder for
            handling image data and transforms for image transformations.
          </li>
          <li>
            <CodeWord>Calculating Mean and Standard Deviation:</CodeWord> It
            calculates the mean and standard deviation of the training dataset.
            This is often done as a preprocessing step for normalization, which
            helps in stabilizing the training process. The mean and standard
            deviation are computed separately for each channel (RGB), then
            averaged across all images in the dataset.
          </li>
          <li>
            <CodeWord>Data Augmentation Configuration:</CodeWord> Data
            augmentation is a technique used to artificially increase the size
            of the training dataset by applying various transformations to the
            images. This helps in improving the generalization of the model and
            reducing overfitting. In this code, data augmentation is configured
            differently for the training, validation, and test datasets.
            <Underline>
              'train': For the training dataset, a series of transformations are
              applied:
            </Underline>
            <List className="pt-10">
              <li>
                transforms.Grayscale(num_output_channels=3): Converts the image
                to grayscale. num_output_channels=3 ensures that the image
                remains in RGB format after conversion.
              </li>
              <li>
                torchvision.transforms.RandomPerspective(distortion_scale=0.6,
                p=1.0): Applies a random perspective transformation to the
                image.
              </li>
              <li>
                torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET):
                Applies auto augmentation techniques based on the IMAGENET
                policy.
              </li>
              <li>
                transforms.ToTensor(): Converts the image to a PyTorch tensor.
              </li>
            </List>
          </li>
          <li>
            <CodeWord>'val':</CodeWord> Similar transformations as the 'train'
            set but without normalization.
          </li>
          <li>
            <CodeWord>'test':</CodeWord> Only applies transforms.ToTensor() to
            convert images to PyTorch tensors, without any additional
            augmentation or normalization.
          </li>
          <li>
            <CodeWord>'cust_test':</CodeWord> Similar to 'test', only applies
            transforms.ToTensor().
          </li>
        </List>
        <CodeBlock file={model3}></CodeBlock>
        <SubHeading id="dense">5. DenseNet</SubHeading>
        <List>
          <li>
            <CodeWord>Loading the DenseNet-121 Model:</CodeWord>
            The code initializes a DenseNet-121 model pre-trained on ImageNet.
            DenseNet (Densely Connected Convolutional Networks) is a type of
            convolutional neural network (CNN) architecture known for its dense
            connections between layers. These connections facilitate feature
            reuse and enhance gradient flow, leading to improved training
            performance.
          </li>
          <li>
            <CodeWord>Freezing Base Layers:</CodeWord>
            By setting `requires_grad` to `False` for the parameters in the
            `features` section of the DenseNet-121 model, the code freezes these
            layers during training. Freezing prevents the weights in these
            layers from being updated during backpropagation. This strategy is
            often employed when using pre-trained models to leverage the learned
            features while training only the classifier layers.
          </li>
          <li>
            <CodeWord>Recreating the Classifier Layer:</CodeWord>
            The existing classifier of the DenseNet-121 model is replaced with a
            new one. This new classifier typically includes layers for adapting
            the features learned by the convolutional base to the specific task
            at hand. In this case, a dropout layer is added for regularization,
            followed by a fully connected linear layer mapping the features to
            the number of output classes. The `class_names` variable determines
            the number of output classes.
          </li>
          <li>
            <CodeWord>Model Initialization:</CodeWord>
            <List className="pt-10">
              <li>
                <CodeWord>Loss Function (criterion_Dense):</CodeWord> The code
                initializes the loss function for training the model. Here, it
                uses the CrossEntropyLoss, which is suitable for multi-class
                classification tasks. This loss function computes the
                cross-entropy loss between the predicted probabilities and the
                ground truth labels.
              </li>
              <li>
                <CodeWord>Optimizer (optimizer_Dense):</CodeWord> Stochastic
                Gradient Descent (SGD) is chosen as the optimizer with a
                learning rate of 0.005 and momentum of 0.9. The optimizer
                updates the model parameters based on the gradients computed
                during backpropagation, aiming to minimize the loss function.
              </li>
              <li>
                <CodeWord>
                  Learning Rate Scheduler (exp_lr_scheduler_Dense):
                </CodeWord>{" "}
                A learning rate scheduler adjusts the learning rate during
                training to improve convergence. Here, a step scheduler is
                employed, which decreases the learning rate by a factor of 0.1
                every 7 epochs. This technique helps to fine-tune the learning
                process and potentially achieve better performance.
              </li>
            </List>
          </li>
        </List>
        <CodeBlock file={model4}></CodeBlock>
        <SubHeading id="efficient">6. EfficientNet</SubHeading>
        <List>
          <li>
            <CodeWord>Import Statements:</CodeWord>The code likely imports
            necessary libraries, but it's not explicitly mentioned. The
            torchinfo.summary function is imported from the torchinfo module.
            The EfficientNet_B0_weights and EfficientNet_B0 models are expected
            to be imported from torchvision.models module. Additionally,
            standard PyTorch modules (torch, torch.nn, optim, lr_scheduler) are
            needed for model configuration and training.
          </li>
          <li>
            <CodeWord>Loading the EfficientNet-B0 Model:</CodeWord>
            <List className="pt-10">
              <li>
                The code initializes the EfficientNet-B0 model using
                torchvision.models.efficientnet_b0 with pre-trained weights
                specified by EfficientNet_B0_weights. EfficientNet is a family
                of convolutional neural network models developed by Google
                Brain. It achieves state-of-the-art performance with fewer
                parameters compared to other models like ResNet or DenseNet.
              </li>
              <li>
                The model is then moved to the specified device (device),
                typically a GPU for faster computation if available.
              </li>
            </List>
          </li>
          <li>
            <CodeWord>Freezing Base Layers:</CodeWord> Similar to the previous
            example, the code freezes the parameters in the features section
            (the convolutional base) of the EfficientNet-B0 model. By setting
            requires_grad to False, these layers are not updated during
            training, retaining the pre-trained knowledge. This strategy is
            commonly used when fine-tuning models on specific tasks to prevent
            overfitting and reduce computational cost.
          </li>
          <li>
            <CodeWord>Recreating the Classifier Layer:</CodeWord> The existing
            classifier of the EfficientNet-B0 model is replaced with a new one.
            The new classifier consists of a dropout layer followed by a fully
            connected linear layer. The dropout layer (torch.nn.Dropout) with
            p=0.2 adds regularization by randomly setting a fraction of input
            units to zero during training to prevent overfitting. The linear
            layer (torch.nn.Linear) maps the features from the EfficientNet-B0
            model to the number of output classes specified by len(class_names).
          </li>
          <li>
            <CodeWord>Model Initialization:</CodeWord>
            <List className="pt-10">
              <li>
                <CodeWord>Loss Function (criterion_EfficientNet_B0):</CodeWord>{" "}
                The code initializes the loss function for training the model.
                Similar to the previous example, it uses the CrossEntropyLoss,
                suitable for multi-class classification tasks.
              </li>
              <li>
                <CodeWord>Optimizer (optimizer_EfficientNet_B0):</CodeWord>{" "}
                Stochastic Gradient Descent (SGD) with a learning rate of 0.005
                and momentum of 0.9 is chosen as the optimizer. The optimizer
                updates the model parameters based on the gradients computed
                during backpropagation to minimize the loss.
              </li>
              <li>
                <CodeWord>
                  Learning Rate Scheduler (exp_lr_scheduler_EfficientNet_B0):
                </CodeWord>
                Learning Rate Scheduler (exp_lr_scheduler_EfficientNet_B0): A
                step scheduler is employed, which decreases the learning rate by
                a factor of 0.1 every 7 epochs. This technique helps in
                fine-tuning the learning process and potentially achieving
                better performance.
              </li>
            </List>
          </li>
        </List>
        <CodeBlock file={model5}></CodeBlock>
        <SubHeading id="mobile">7. MobileNet</SubHeading>
        <List>
          <li>
            <CodeWord>Import Statements:</CodeWord> Presumably, necessary
            libraries are imported, including the torchinfo.summary function for
            summarizing the model's architecture and parameter count.
            Additionally, the torchvision.models module is imported to access
            the MobileNetV3 model. Standard PyTorch modules (torch, torch.nn,
            optim, lr_scheduler) are needed for model configuration and traini
          </li>
          <li>
            <CodeWord>oading the MobileNetV3 Model:</CodeWord>
            <List className="pt-10">
              <li>
                The code initializes a MobileNetV3 model with the large variant
                using torchvision.models.mobilenet_v3_large. The
                weights='DEFAULT' argument loads the pre-trained weights if
                available.
              </li>
              <li>
                The model is then moved to the specified device (device),
                typically a GPU for faster computation if available.
              </li>
            </List>
          </li>
          <li>
            <CodeWord>Freezing Base Layers:</CodeWord> The code iterates through
            the parameters in the features section (the convolutional base) of
            the MobileNetV3 model and sets requires_grad to False. This
            effectively freezes these layers during training, preventing them
            from being updated and retaining the pre-trained knowledge. This
            strategy is commonly used to prevent overfitting and reduce
            computational cost.
          </li>
          <li>
            <CodeWord>Recreating the Classifier Layer:</CodeWord>
            <List className="pt-10">
              <li>
                The existing classifier of the MobileNetV3 model is replaced
                with a new one. The new classifier consists of a dropout layer
                followed by a fully connected linear layer.
              </li>
              <li>
                The dropout layer (torch.nn.Dropout) with p=0.2 adds
                regularization by randomly setting a fraction of input units to
                zero during training to prevent overfitting.
              </li>
              <li>
                The linear layer (torch.nn.Linear) maps the features from the
                MobileNetV3 model to the number of output classes specified by
                len(class_names).
              </li>
            </List>
          </li>
          <li>
            <CodeWord>Model Initialization:</CodeWord>
            <List className="pt-10">
              <li>
                <CodeWord>Loss Function (criterion_MobileNet_V3):</CodeWord> The
                code initializes the loss function for training the model. Here,
                it uses the CrossEntropyLoss, suitable for multi-class
                classification tasks.
              </li>
              <li>
                <CodeWord>Optimizer (optimizer_MobileNet_V3):</CodeWord>{" "}
                Stochastic Gradient Descent (SGD) with a learning rate of 0.005
                and momentum of 0.9 is chosen as the optimizer. The optimizer
                updates the model parameters based on the gradients computed
                during backpropagation to minimize the loss.
              </li>
              <li>
                <CodeWord>
                  Learning Rate Scheduler (exp_lr_scheduler_MobileNet_V3):
                </CodeWord>{" "}
                A step scheduler is employed, which decreases the learning rate
                by a factor of 0.1 every 7 epochs. This technique helps in
                fine-tuning the learning process and potentially achieving
                better performance.
              </li>
            </List>
          </li>
        </List>
        <CodeBlock file={model6}></CodeBlock>
        <SubHeading id="training">8. Traninig M0del</SubHeading>
        <CodeBlock file={model7}></CodeBlock>
        <SubHeading id="ensembler">9. Ensembler</SubHeading>
        <CodeBlock file={model8}></CodeBlock>
      </Block>
    ),
  },
];
function Docs() {
  const [showIndices, setShowIndices] = useState(true);
  const toggle = () => {
    setShowIndices((current) => !current);
  };
  return (
    <div className="flex bg-black h-screen">
      <div
        className={twMerge(
          "space-y-6 bg-black rounded-tr-xl h-full overflow-y-scroll relative snap-y scroll-pt-40",
          showIndices && "w-3/4"
        )}
      >
        <div className="top-0 sticky flex flex-col items-end">
          <div className="bg-gray-200/5 backdrop-blur-sm py-6 w-full h-20">
            <p className="px-20 2xl:px-40 font-extrabold font-libre text-3xl text-gray-200">
              DOCUMENTATION
            </p>
          </div>
          <Button
            className="bg-gray-200/5 hover:bg-gray-200/5 ml-full px-1 rounded-none rounded-bl-lg w-8"
            onClick={toggle}
          >
            <ChevronsLeft
              className={twMerge("text-gray-200", showIndices && "rotate-180")}
            />
          </Button>
        </div>
        <div className="space-y-12 px-20 2xl:px-40">
          {docBlocks.map((cod) => cod.code)}
        </div>
      </div>
      {showIndices && <Indices setShowIndices={setShowIndices} />}
    </div>
  );
}

export default Docs;
