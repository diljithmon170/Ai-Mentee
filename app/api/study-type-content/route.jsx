import { STUDY_TYPE_CONTENT_TABLE } from "/configs/schema";
import { db } from "/configs/db";
import { inngest } from "/inngest/client";
import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const { chapter, courseId, type } = await req.json();

    if (!chapter || !courseId || !type) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    const PROMPT =
      type === "Flashcard"
        ? `Generate a ${type} on the topic: "${chapter}" in JSON format with front-back content, Maximum 15 entries.`
        : type === "Quiz"
        ? `Generate a Quiz on the topic: "${chapter}" with questions, options, and correct answers in JSON format (Maximum 10 entries).`
        : `
        Create **15 question-and-answer pairs** based on the following topics: 
        ${chapter}

        The Answer should be at least of 10 lines	

### **Output Requirements**
1. **Questions and Answers:**
   - Each question must be concise and directly address one of the topics above.
   - Each answer should be **detailed and explanatory**, providing:
     - A clear explanation of the concept.
     - Examples or scenarios illustrating the answer where applicable.
     - Practical tips or best practices.


### **Output Example for a Question**
emoji: A relevant emoji to visually represent the chapter.
content (string): Detailed content for the topic written in Md format, and ready for rendering in a React.js component.

EXAMPLE OUTPUT
{
  "questions": [
    {
      "question": "💻 Explain the concept of Java Streams and its advantages over traditional loops.",
      "answer": "Java Streams provide a declarative way to process collections of data. Unlike traditional loops (for, while), which are imperative and specify *how* to process data, Streams declare *what* needs to be done.  This functional approach enhances code readability and maintainability.  \n\nAdvantages:\n* **Conciseness:** Streams often require fewer lines of code compared to loops.\n* **Readability:** The declarative style improves code understanding.\n* **Parallelism:** Streams easily support parallel processing, significantly speeding up computations on large datasets.  This is achieved using 'parallelStream()'.\n* **Lazy Evaluation:**  Intermediate operations in a stream pipeline are not executed until a terminal operation (like 'collect()', 'forEach()', 'reduce()') is called. This optimizes performance by avoiding unnecessary computations.\n* **Composability:**  Multiple stream operations can be chained together, creating elegant and expressive data pipelines.\n\nExample:\n 'java\nList<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);\nint sum = numbers.stream().filter(n -> n % 2 == 0).map(n -> n * 2).reduce(0, Integer::sum);\nSystem.out.println(sum); // Output: 60\n' \nThis code calculates the sum of even numbers after doubling them using streams.  The equivalent using a loop would be significantly longer and less readable.\n\nBest Practices:\n* Use streams for operations involving collections.\n* Leverage parallel streams for performance gains on large datasets.\n* Understand the difference between intermediate and terminal operations.\n* Avoid excessive chaining to prevent code complexity."
    },
    {
      "question": "🌱 What are Spring beans and dependency injection?  Illustrate with an example.",
      "answer": "In Spring Framework, a bean is an object that is instantiated, assembled, and managed by the Spring IoC (Inversion of Control) container.  Dependency Injection (DI) is a design pattern where dependencies are provided to a class instead of the class creating them itself. Spring manages this injection process automatically.\n\nHow it Works:\nThe Spring container reads configuration metadata (usually from XML, annotations, or Java-based configurations) to determine which beans to create and how to wire them together.  DI eliminates hard-coded dependencies, making code more flexible, testable, and maintainable.\n\nExample:\nLet's say we have a 'Car' class that depends on an 'Engine' class:\n 'java\n// Car class\npublic class Car {\n    private Engine engine;\n\n    // Constructor-based injection\n    public Car(Engine engine) {\n        this.engine = engine;\n    }\n    // ... other methods ...\n}\n\n// Engine class\npublic class Engine {\n    // ...\n}\n' \nIn the Spring configuration, we define the beans and specify the dependency:\n' xml\n<bean id=\"engine\" class=\"com.example.Engine\" />\n<bean id=\"car\" class=\"com.example.Car\">\n    <constructor-arg ref=\"engine\" />\n</bean>\n' \nSpring automatically creates an 'Engine' bean and injects it into the 'Car' bean's constructor.  This removes the responsibility of creating the 'Engine' from the 'Car' class.\n\nBest Practices:\n* Use constructor injection whenever possible for better control and testability.\n* Use annotations ('@Component', '@Autowired', '@Service', etc.) for simpler configuration.\n* Understand the different scopes of beans (singleton, prototype, etc.)."
    },
    {
      "question": "🌐 Describe the key characteristics of a microservices architecture.",
      "answer": "Microservices architecture is an approach to software development where a large application is structured as a collection of small, autonomous services. Each service focuses on a specific business function and communicates with others via lightweight mechanisms, often RESTful APIs. \n\nKey Characteristics:\n* **Independent Deployability:** Each microservice can be deployed, scaled, and updated independently of others. This allows for faster release cycles and better fault isolation.\n* **Decentralized Governance:** Teams can choose their own technologies and development processes for each service. This promotes innovation and faster development.\n* **Bounded Contexts:** Each microservice represents a specific business domain or sub-domain. This promotes modularity and reduces complexity.\n* **Resilience:** Failure of one microservice should not bring down the entire application. This is achieved through techniques like circuit breakers and fault tolerance.\n* **Data Decentralization:**  Each microservice typically owns its own data.  This can lead to data consistency challenges, requiring careful management.\n* **Technology Diversity:** Teams can choose the technologies best suited for each service's requirements.\n\nChallenges:\n* **Increased Complexity:** Managing many services can be challenging.  Requires robust monitoring and logging.\n* **Distributed Transactions:** Maintaining data consistency across multiple services can be difficult.\n* **Inter-service Communication:** Choosing and managing communication protocols is important.\n\nExample:\nAn e-commerce application might have separate microservices for user accounts, product catalog, order management, and payment processing. Each can be developed, deployed, and scaled independently."
    },
    {
      "question": "🔗 Explain the principles of RESTful API design.",
      "answer": "REST (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs use HTTP methods (GET, POST, PUT, DELETE) to interact with resources.  Key principles:\n\n* **Client-Server:**  The client and server are independent and can be developed and updated separately.\n* **Stateless:** Each request from the client to the server must contain all the information necessary to understand the request. The server doesn't maintain client context between requests.\n* **Cacheable:** Responses from the server should be cacheable to improve performance.  HTTP headers like 'Cache-Control' manage caching.\n* **Uniform Interface:** A consistent interface should be used for interacting with resources.  This promotes simplicity and interoperability.\n    * **Resources are identified by URIs:** (e.g., '/users/{id}').\n    * **Standard HTTP methods are used:** GET (retrieve), POST (create), PUT (update), DELETE (delete).\n    * **Representations of resources are returned:** (e.g., JSON or XML).\n    * **Hypermedia controls:**  Links in responses guide clients to related resources (HATEOAS).\n* **Layered System:**  The client typically interacts with a load balancer or API gateway which forwards the request to the actual microservices.\n* **Code on Demand (optional):**  The server can extend client functionality by transferring executable code.\n\nBest Practices:\n* Use meaningful URLs.\n* Employ proper HTTP status codes.\n* Use consistent data formats (JSON is common).\n* Implement appropriate error handling and logging.\n* Design for scalability and maintainability.\n* Use versioning to manage API changes."
    },
    {
      "question": "⚛️  Discuss the key features and benefits of React.js for frontend development.",
      "answer": "React.js is a popular JavaScript library for building user interfaces (UIs). It's known for its component-based architecture, virtual DOM, and JSX syntax.\n\nKey Features:\n* **Component-Based Architecture:** UIs are built using reusable components, improving code organization and maintainability.  Components manage their own state and render their own output.\n* **Virtual DOM:** React uses a virtual DOM (Document Object Model) to optimize updates.  Changes are first applied to the virtual DOM, then efficiently updated in the actual DOM, improving performance.\n* **JSX:** JSX (JavaScript XML) allows writing HTML-like code within JavaScript, making it easier to create UI elements.\n* **One-way Data Binding:**  Data flows in a single direction, simplifying state management and reducing complexity.\n* **Large Community and Ecosystem:** Abundant resources, third-party libraries, and tools are available.\n\nBenefits:\n* **Increased Productivity:** Component reusability and simplified development process leads to faster development.\n* **Improved Performance:** Virtual DOM optimizes rendering for a smoother user experience.\n* **Easier Debugging and Maintenance:**  Modular design and component-based architecture improve code readability and maintainability.\n* **Code Reusability:** Components can be used across different projects.\n* **SEO-Friendly:**  React applications can be rendered server-side (SSR) for better SEO.\n\nExample:\nA simple React component:\n 'javascript\nfunction Welcome(props) {\n  return <h1>Hello, {props.name}</h1>;\n}\n' \nThis component takes a 'name' prop and renders a greeting message."
    },
    {
      "question": "🗄️ Explain the concept of database normalization and its importance.",
      "answer": "Database normalization is a process of organizing data to reduce redundancy and improve data integrity.  It involves breaking down a database into two or more tables and defining relationships between the tables.\n\nImportance:\n* **Reduces Data Redundancy:**  Eliminating duplicate data saves storage space and improves data consistency.\n* **Enhances Data Integrity:** Ensures that data is accurate and reliable.  Changes to data are made in only one location.\n* **Improves Data Management:** Simplifies queries and data modification.  Easier to manage and maintain.\n* **Minimizes Data Anomalies:** Prevents update, insertion, and deletion anomalies.  These are inconsistencies that occur when data is not properly structured.\n\nNormal Forms:\nDatabase normalization is often described in terms of normal forms (1NF, 2NF, 3NF, BCNF, etc.).  Each normal form represents a level of data organization.\n* **1NF (First Normal Form):**  Eliminates repeating groups of data within a table. Each column should contain atomic values (single values).\n* **2NF (Second Normal Form):**  Must be in 1NF and eliminate redundant data that depends on only part of the primary key (in tables with composite keys).\n* **3NF (Third Normal Form):** Must be in 2NF and eliminate transitive dependency; values should depend directly on the primary key, not indirectly via other non-key attributes.\n\nExample:\nA poorly normalized table (students and courses):\n| StudentID | Name  | Course1 | Course2 |\n|---|---|---|---| \n| 1 | John  | Math   | Science |\n| 2 | Jane  | Science | History |\n\nNormalized tables:\n**Students:**\n| StudentID | Name |\n|---|---| \n| 1 | John  |\n| 2 | Jane  |\n\n**Courses:**\n| CourseID | CourseName |\n|---|---| \n| 1 | Math   |\n| 2 | Science |\n| 3 | History |\n\n**StudentCourses:**\n| StudentID | CourseID |\n|---|---| \n| 1 | 1 |\n| 1 | 2 |\n| 2 | 2 |\n| 2 | 3 |"
    },
    {
      "question": "🧱 Explain the concept of ACID properties in database transactions.",
      "answer": "ACID properties are a set of four characteristics that guarantee database transactions are processed reliably. These properties ensure data consistency and integrity even in the event of failures:\n\n* **Atomicity:** A transaction is treated as a single, indivisible unit of work. Either all operations within the transaction are completed successfully, or none are.  If any part fails, the entire transaction is rolled back to its previous state.\n* **Consistency:**  A transaction must maintain the database's integrity constraints. The database must remain in a valid state before and after a transaction. It should not violate any rules or constraints set up (e.g., data types, foreign key constraints).\n* **Isolation:**  Concurrent transactions must be isolated from each other.  Changes made by one transaction are not visible to other transactions until the first transaction is completed. This avoids data corruption caused by race conditions.\n* **Durability:**  Once a transaction is committed, the changes are permanently saved to the database, even if a system failure occurs.  The data is persisted, even through crashes or power outages.\n\nExample:\nImagine transferring money from one bank account to another.  ACID properties ensure:\n* **Atomicity:**  Either both debiting and crediting occur, or neither does. No partial transfers are possible.\n* **Consistency:**  The total amount of money remains the same after the transaction.\n* **Isolation:**  Other users cannot see the intermediate state where one account is debited but the other is not yet credited.\n* **Durability:** The transaction is permanently recorded, even if the system crashes immediately after it is completed."
    },
    {
      "question": "⚙️ What is the purpose of a connection pool in database management?",
      "answer": "A database connection pool is a caching mechanism that improves database application performance by reusing connections instead of repeatedly creating and closing them.  Creating a database connection is a resource-intensive operation, so pooling minimizes the overhead and makes the application more responsive. \n\nHow it Works:\nThe pool maintains a set of active and idle database connections. When an application needs a connection, it requests one from the pool.  If an idle connection is available, it's reused; otherwise, a new connection is created (up to a maximum limit). When the application is finished with the connection, it returns it to the pool, where it becomes available for reuse. Connections that remain idle for a long time are eventually closed to manage resources.\n\nBenefits:\n* **Improved Performance:** Reduced overhead associated with connection creation and destruction leads to faster application response times.\n* **Resource Management:** Efficiently manages database connections, preventing resource exhaustion.\n* **Scalability:**  Facilitates scaling of applications by providing a readily available pool of connections to handle increased traffic.\n* **Connection Lifetime Management:**  Provides controls for connection timeout and recycling, preventing issues related to stale connections. \n\nCommon Pooling Libraries:\n* HikariCP (popular choice for its speed and efficiency).\n* c3p0\n* Apache Commons DBCP"
    },
    {
      "question": "💡 Explain the concept of Java Generics and its benefits.",
      "answer": "Generics in Java allow you to write type-safe code that can work with various data types without losing type information at compile time.  Before generics, collections like 'ArrayList' stored 'Object' types, requiring explicit casting, which could lead to runtime 'ClassCastException' errors. Generics solved this by introducing type parameters.\n\nBenefits:\n* **Type Safety:**  Compile-time type checking prevents 'ClassCastException' errors.  The compiler ensures only compatible types are used.\n* **Improved Code Readability:**  Generics make code clearer by explicitly indicating the types being used.\n* **Reduced Code Duplication:**  You can write a single generic class or method that works with different types, eliminating the need to write separate versions for each type.\n* **Enhanced Maintainability:**  Changes to types are automatically propagated throughout the code, reducing the risk of errors.\n\nExample:\n 'java\nList<String> names = new ArrayList<>(); // List of Strings\nnames.add(\"Alice\");\nnames.add(\"Bob\");\nString name = names.get(0); // No casting needed\n' \nWithout generics, 'names' would be 'List<Object>', and getting elements would require casting: 'String name = (String) names.get(0);' - which is prone to errors if the list contains non-String objects.\n\nBest Practices:\n* Use generics whenever possible to improve type safety and code clarity.\n* Use wildcards ('? extends T', '? super T') for more flexible type handling.\n* Avoid raw types (e.g., 'ArrayList' instead of 'ArrayList<String>'), as they defeat the purpose of generics."
    },
    {
      "question": "🛠️ Describe different types of Spring dependency injection.",
      "answer": "Spring Framework provides several ways to inject dependencies into beans:\n\n* **Constructor Injection:** Dependencies are passed as arguments to a class's constructor.  This approach is generally preferred as it ensures all required dependencies are provided.  It promotes immutability and better testability because dependencies are immediately available when the bean is created.\n* **Setter Injection:** Dependencies are injected using setter methods. This allows for setting dependencies after the bean is created.  It's useful for optional dependencies but can lead to objects not being fully configured if dependencies are not set appropriately.\n* **Field Injection:** Dependencies are directly injected into a bean's fields using '@Autowired' annotation.  This approach is generally discouraged because it reduces readability and testability; it bypasses constructor and setter validation, and the object might not be in a valid state until all fields are injected. \n\nExample (Constructor Injection):\n 'java\npublic class Car {\n    private final Engine engine;\n\n    @Autowired // Spring handles this injection\n    public Car(Engine engine) {\n        this.engine = engine;\n    }\n}\n' \nExample (Setter Injection):\n 'java\npublic class Car {\n    private Engine engine;\n\n    @Autowired // Spring handles this injection\n    public void setEngine(Engine engine) {\n        this.engine = engine;\n    }\n}\n' \nChoosing the right method depends on the specifics of the application. Constructor injection is generally recommended for mandatory dependencies, while setter injection is suitable for optional ones."
    },
    {
      "question": "🗂️ Explain different types of database relationships.",
      "answer": "Database relationships define how data in different tables is connected.  The most common types are:\n\n* **One-to-One:**  A single record in one table is related to a single record in another table. Example: A 'person' table and a 'passport' table, where one person has at most one passport.\n* **One-to-Many:** A single record in one table can be related to multiple records in another table.  Example: A 'customer' table and an 'order' table, where one customer can have multiple orders.\n* **Many-to-One:**  Multiple records in one table can be related to a single record in another table. This is the inverse of a one-to-many relationship. Example: Multiple 'employees' can report to a single 'manager'.\n* **Many-to-Many:** Multiple records in one table can be related to multiple records in another table. Example:  'students' and 'courses', where multiple students can take multiple courses. This relationship usually requires a junction table (or bridge table) to represent the association.\n\nImplementation:\nThese relationships are implemented using foreign keys. A foreign key in one table references the primary key of another table, creating the link between the records.  For many-to-many relationships, the junction table has foreign keys to both participating tables."
    },
    {
      "question": "🔄 Explain the concept of state management in React.",
      "answer": "State management in React refers to how you handle and update data that changes over time in your application. Effectively managing state is crucial for building complex React apps that are predictable, maintainable, and performant.  Without a well-defined strategy, large React apps become difficult to debug and update.\n\nApproaches to State Management:\n* **Local State (useState):** For simple components, you can manage state directly using the 'useState' hook.  The state is contained within the component, making it easy to manage, but not ideal for complex apps where state needs to be shared across many components.\n* **Component State (lifting state up):**  For state that needs to be shared across several child components, you can ‘lift state up’ to the nearest common ancestor component, passing the state and update functions down as props. While this works for moderately complex applications, it becomes cumbersome as the app grows.\n* **Context API:** Provides a way to share state globally across the application.  Useful for data that is used across many components, but it can become difficult to manage in very large apps.\n* **Redux (and other state management libraries):** Redux, Zustand, Jotai, and Recoil are more advanced state management libraries that provide features like centralized state management, unidirectional data flow, and time-travel debugging. They're suitable for large-scale applications but add complexity.  Each library has its own approach; Redux is the most established but can feel overly complex for some projects.  Zustand is a popular alternative that's lightweight and easy to learn.\n\nChoosing the Right Approach:\nStart with local state and lifting state up for smaller applications.  As your app grows, consider the Context API or a dedicated state management library like Redux or Zustand to maintain scalability and maintainability."
    },
    {
      "question": "🌐  Explain the role of an API Gateway in a microservices architecture.",
      "answer": "In a microservices architecture, an API Gateway acts as a single entry point for all client requests.  It sits in front of the multiple microservices that make up the application and provides several key functionalities:\n\n* **Request Routing:** The API Gateway routes incoming requests to the appropriate microservices based on the request's path and other criteria.  This hides the internal structure of the application from clients.\n* **Authentication and Authorization:** The API Gateway can handle authentication and authorization, securing access to the microservices. It centralizes security, rather than having each microservice implement its own security logic.\n* **Request Transformation:** The Gateway can transform requests or responses to match the needs of the clients or the microservices.  It might adjust data formats, aggregate data from multiple services, or filter data.\n* **Load Balancing:** Distributes requests across multiple instances of a microservice to ensure high availability and performance.\n* **Rate Limiting:** Prevents abuse by limiting the number of requests a client can make within a given time period.\n* **Monitoring and Logging:** The Gateway can monitor and log all incoming and outgoing traffic, providing insights into application performance and usage.\n* **Caching:** Caches frequently accessed data to improve performance and reduce load on the microservices.\n\nBenefits:\n* **Simplified Client Interactions:** Clients interact with a single endpoint instead of many individual microservices.\n* **Improved Security:** Centralized authentication and authorization enhance security.\n* **Enhanced Performance:** Request transformation, load balancing, and caching improve performance.\n* **Reduced Complexity:** Abstracts the internal complexity of the microservices from the clients."
    }
  ]
}


2. Content Formatting:
Give me in .md format
{
  "questions": [
    {
      "question": "",
      "answer": ""
    }
}

**IMPORTANT**
There should be an emoji
Give me in .md format



 7. **Additional Notes:**  
   - **IMPORTANT** There should be an emoji
   - Every Content should be in detail and explained properly
   - Each 'content' field should use simple and concise language suitable for study notes.  
   - Ensure that topics include clear definitions, key points, and, where appropriate, examples or sample code.  
   - All generated content should be focused on clarity and exam preparation, with minimal redundancy.  

 8. **Avoid Common Errors:**  
   - Double-check for mismatched brackets, missing fields, or improperly formatted strings.  
        
        `;

    console.log("Prompt:", PROMPT);

    // Insert into the database
    const result = await db
      .insert(STUDY_TYPE_CONTENT_TABLE)
      .values({
        courseId: courseId,
        type: type,
        chapter: chapter, // Add chapter for better traceability
      })
      .returning({ id: STUDY_TYPE_CONTENT_TABLE.id });

    console.log("Inserted Content ID:", result);

    // Trigger the external task
    await inngest.send({
      name: "studyType.content",
      data: {
        studyType: type,
        prompt: PROMPT,
        courseId: courseId,
        recordId: result[0]?.id,
      },
    });

    return NextResponse.json({ id: result[0]?.id });
  } catch (error) {
    console.error("Error in POST /api/study-type-content:", error.message);
    return NextResponse.json(
      { error: "Failed to generate study material. Please try again." },
      { status: 500 }
    );
  }
}
