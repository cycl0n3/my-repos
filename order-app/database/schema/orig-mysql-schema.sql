/*
SQLyog Community v13.2.0 (64 bit)
MySQL - 10.11.2-MariaDB : Database - order-app
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`order-app` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci */;

USE `order-app`;

/*Table structure for table `brands` */

DROP TABLE IF EXISTS `brands`;

CREATE TABLE `brands` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `url` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  `image` text DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `brands` */

insert  into `brands`(`id`,`name`,`url`,`description`,`image`,`created_at`,`updated_at`) values 
(1,'HP','https://www.hp.com','Description for HP brand','data:image/png;base64,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','2023-12-18 11:39:06','2023-12-18 11:39:06'),
(2,'Apple','https://www.apple.com','Description for Apple brand','data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAL0AygMBIgACEQEDEQH/xAAbAAEBAAIDAQAAAAAAAAAAAAAABgQFAgMHAf/EADgQAQACAgACBQoEBAcAAAAAAAABAgMEBREGEiFBYRMiMVFxgZGxwdEyUqHhFELw8RYjQ1NicrL/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8A9xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABr+L8Ww8Nxx1vPzWjzMcT+s+qAbAQO7xbd3bT5XNatJ/06dlf396o6MUmnB8XPs61rTHxBtgAAAAAAAAAAAAAAAAAAAAAdO3sU1dbJnyfhx15z4+Dz7a2Mm1sXz5p53vPOfDwVfS7LNOGUpE9mTLET7IiZ+cQltTU2NzJ1NbFbJPfMeiPbPcDjq6+Ta2MeDDHO955R4eL0PXw118GPDj/DjrFY9zX8E4PThtJveYvsWjla0eiI9UNoAAAAAAAAAAAAAAAAADp29jHqa98+aeVKRznx8AfdnZw6uKcuxkrSkd8p7b6Vdsxp6/OPz5Z+kfdpOJb+biGxOXNPKI/BSPRWGIDdf4m4h1ufLB7OpPL5s/U6U0tMV3ME0/54+2Ph/dLAL6N7hu3jibZ9a9I7eWSY7PHlPoYe10h0NWvU148tMeiMccqx7/tzRoDd5+k29kmfJVxYq93KOtPxn7OGPpJxGk87WxX8LU+3JpwFbodJ8OW0U3MfkZn+eJ51/Zvq2resWrMTWY5xMT2S80UvRHLuc74+pNtOP5pn8NvD7f1IU4AAAAAAAAAAAAACW6X7c2zYtSs+bWOvf2z6P68VSgeN5PK8W2rT3ZJr8Oz6AwQAAAAAAAZvCdC3EN2uGOcUjzslo7qrzDix4MVcWKsVpWOUVjuaPohgimllzzHnZL8ufhH7zLfgAAAAAAAAAAAAAAPPOJdnEdvn/vX/APUvQ0Fx7H5Li+1X1363xjn9QYAAAAAAAALPopeLcJiI/kyWifn9W5SfRLdjFsZNXJPKMvbT/tHd74+SsAAAAAAAAAAAAAAASfTDX6u1h2IjsyU6s+2P7/orGu4/p/xnDMlaxzyY/Pp7Y/bmCEAAAAAAAB9raa2i1ZmLRPOJjulX8G4/i2aVw7lq488dnWnsrf7T4I93aevbb2sWvT05LcvZHfPwB6MOOOlcdK0pHKtYiIj1Q5AAAAAAAAAAAAAAAh+kPD50d2bUj/Jy87U8J74at6FxHSx7+pfBk7OfbW35Z7pQe1r5dTPfBnr1b1nt8fGPAHSAAAAAAqeifD5pS29ljttHVxc/V3y0/BeGX4jtRExMYKduS30jxXVK1pStKRFa1jlER3QD6AAAAAAAAAAAAAAAAweKcL1+JY4jLE1yV/Dkr6Y+8M4BIX6L7sX5Uy4LV/NMzH6cmbpdF8VJi25mnJP5KdkfH0/JRAMTHwzRxV6tNTDy8aRM/GXDY4Pw/Yryvq46z66R1Z/RnAJfa6K3i3PU2KzX8uWOUx74+zjrdFc03j+L2KVp6sXOZn3zHYqgHTq62HUw1w69IpSvd63cAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/Z','2023-12-18 11:39:06','2023-12-18 11:39:06'),
(3,'Dell','https://www.dell.com','Description for Dell brand','data:image/png;base64,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','2023-12-18 11:39:06','2023-12-18 11:39:06'),
(4,'Samsung','https://www.samsung.com','Description for Samsung brand','data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAKgAswMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAABgcBAgUECAP/xABEEAABAwMBBQUDBQwLAAAAAAABAAIDBAURBgcSEyExQVFxgZEyYaEUIrGywhUWIzQ2QlJjdKKz0SUnN1ViZHJzdZKU/8QAGgEBAQEBAQEBAAAAAAAAAAAAAAECAwQFBv/EACsRAQACAgECBAYBBQAAAAAAAAABAgMREgQhEyIxYUFRcZHR8DIFFFKhwf/aAAwDAQACEQMRAD8AupERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBcWo1Zp2mmMM97t7JAcFpnbyUb2z3GqodKwxUsjo21dSIpXNOCWbrnbufeQPIFQTQOj7Vqez3J0tZKy5QkiGCMgbo3chxb+cCcjyXvw9JS2Hxck6j2crXmLcYXlR1lNXQiaiqIqiI9HxPDh6hfrI9kUbpJXtYxoy5zjgAd5KqLZTY9S2TUZNdbaqloKiFwm3yNzeHNp5Hr2eamO1Ws+SaGuAzzn3IB795wz8Mrnk6aK54x1tuJ13ai/l3MJD917X/eVF/6GfzXqhlinibLBIySN3svY4OB8CF8xw2oP0zVXXcH4KtigzjoCx5Px3PRXJsYqxUaLbTg86Spkjx3ZO/8AbXbqehrhx8623qdM0yTadTCaVNbSUpaKqqggLubRLK1ufDJWkFxoaiQRU9bTSyHoyOZrifIFVNt3wblZgRnEEv1mqPaQa2y6405Nya2rjhdy/WtLPrK4+gi+DxOXfU9vok5dW1pftTWUtLu/KqmCDe9niyBufDK0huNBPIIoK6llkd0ZHM1xPkCqm2wA3TWlmtTOZ4bG47jJJjPo0LgbNWMg2jUbGABolmYPANd/JKdDFsHicu+t6Jy+bWn0EuPUap0/TTGGe9W9kgOC01Dcg+qjO2e41VDpaGGle6NtXUiKVzTg7m652M+/A8lA9DaOtepbDcpXVcrbpATwaeIgYAblpLT1BOQsYekpbF4uS2o9lteYtxhetNUwVcLZqWeOaJ3svieHNPmFrU1tJSFoq6qCAu9niytZnwyVVeyazalsl+lbcLfU0tvqIHcQSY3eICN08j16habeADVWXIB/BzfS1SOkrPURii24n4rznhy0thtVTPpjUsqIXU4BPGEgLMDqd7otKa4UVU8spaymneBktima4gd+AVXunh/UXUDs+RVv8WVRvYe0DVVXgD8Rd9div9nHDJbf8ZmDn3iPmuiprKWlLRVVUEBd7PFkDc+GSsU9fR1TzHTVlPM8DJbFK1xx34BVVbeADUWXIB+ZN9LVGNCyv07rOyzykNgroWguxjLJRj4PA9FrH0MXweJy79+30Scura0vue4UNNIY6itpopAMlkkzWn0JXoBDgC0ggjII7VQe2ho+/WoJAOaSL6Crmqax9v0lJWxgF9Pb+I3PeI8j4rjl6XhjpaJ/ktb7mY+T97jfLTa3BtxuVJTOPPdlma0+i3t13tt0Djba+mqt3rwZQ4jyCoLQ9iGs9TzU9zrJwTA+pmmbgySEOa3qc9r/AIL97vpq96V1RI6xU1wqRSPbJBURwOO8MA7pLRg9x7CvVP8AT8UW8Pn5tb9mPFt667PoNFqx28xriCMgHB7EXyndz9QWmgvlrlt90AMEnMHeALHDo5p7wqS1Hoa9aWkNyt1Q6qpYDvNrKQlskI73AHI8Ry8FYm2Gy1d301FJQwumko5+K+Ngy4s3SCQO0jIOPFV5ofXrNK2uuoH28VPGkMjDxA3ddu7pDhjpyHxX2OhrljFzxzvv3q4ZJry1P3TDZptBqrtWx2W9lslS9p+T1IGDIQMlrh34BOR3LG3WsDLVaqEH50tQ6YjvDG4+2onsls9XcdWUtyZCRR0RfJJKG4ZvFrmhoPacnp3BdDbXLNVampaeKKR7aalHNrCRlziSPQBdPBx166Ip8tz9f3ScpnH3Lba+JsTuU+7899X8pB9zHtafg13quhsJrPn3egJ7I52jv6tP2VJrNaXv2Sst+4Wyz2t53SOYe9pcMjvyQq82QTTUes4GSQzMZVU8kWXMIGcb4z/1Um3i4c0e8z+/Y1xtV09up/pm0j/LPP7wXG1PF9z7Xoe8tHs0UeSP1Tw8fWK7W3COWW+W3hxSPDaV2S1pP56/XV1tkqNkenJhE8yUrYstDTnDmlp5eOFvBeK4sMT8ZmPvtLRu1mozetuO8MOhpnh3fhrIR9s/FR3QJLdplD76ucfuyKQ7FqOefUNyuNXHIHR0wYHPYRkvd/JnxUf0ZDNHtJonugla0V0vziwgcw8dfNb7RGTH/jSI/wBSnrqfdd+orTb73a5bfdAODL0dvBrmuHRzT3hUnqLRF70pIblbql1TSwnLaykcWviH+IA5HiCR4KxNsFjqrxpyGWhhdPJRTcV0TBlzmFpBwO3HI+SrvRmvfvUs9dbH25tSZZHPaXSbu44tDSHDHMcvpXm6GuWMXLHO+/ereSY5an7ppsy1/VXqrFmvIa+q3C6GpaMcTHVrh3455HXmuZt3/GrN/ty/S1cjY5ZKup1JBchC8UVHG4mcjDXOI3Q0Ht6k8umPBSrbVZKyvoaG40cMkzaQvbMyNu8WtdjDsDsyOfitTXFi6+sV7R/3unmtj7mnv7C6j9jrf4sqjmxD8qqv9hd9di5NDr11FoWbTAo43b7JYxUGXG62Rxc7LcdfnHHNSnYnZKyGrrLxUwSRU7oRDCXtxxMuBJHuG6OfvW8tJxYc037cpnSRPK1dMbd/xiy/6JvpauDqm3uOz7Sd5hyHxMdTOcB0y5zmnyLT6qRbcaaonns5p6eaXdZLnhxl2ObeuF07dZ5brsaZQOie2oFO98bHtIIeyRzm8j4fFYxZYx9Pit7/AJWa7tZXe0q4tu93pLi3l8qtcEpHcSHZHkcq6L7+QNb/AMWf4a+eZ6S5zQsa+hrDw49xmad/IZJx073FfSctCbhph1A47hqKHhZP5pLMJ10Vx1xR8In8GPc7U/sV3/vrrOF7f3Ll3fHiRYWl91rrqw1stBcrhC2qiYHODImOHMZHPC5lkr7ps71HJPXW48QROp5GS5a17SWnLHYwebRzWKiO7bRdTvqKaiLPlJaxz4wTHAwDGXOPLkOfv7AvXbHWc05LxE016sb8sRHq+hqV7pKWF7zlzo2kn3kIt42CONsbfZa0NHgEX5qXrbLxz2q21EnFqLdRyyfpyU7HH1IXsRWJmPQaxxsiYGRMaxg6NaMAeSzgdyyigLGB3LKIMYHcsoiDGMJgdyyiAvNNb6KeTiT0VNI/9J8LSfUhelFYmY9BhrQ1oa0BrR0AGAFlEUHndQ0bpeK6kpzJ+mYm59cL0e5EV3IAkdCiIoM7x7ysIiDDmtcMOaHD3jKNAaMNAA7gMLKICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiD/9k=','2023-12-18 11:39:06','2023-12-18 11:39:06'),
(5,'Lenovo','https://www.lenovo.com','Description for Lenovo brand','data:image/jpeg;base64,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','2023-12-18 11:39:06','2023-12-18 11:39:06');

/*Table structure for table `failed_jobs` */

DROP TABLE IF EXISTS `failed_jobs`;

CREATE TABLE `failed_jobs` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `uuid` varchar(255) NOT NULL,
  `connection` text NOT NULL,
  `queue` text NOT NULL,
  `payload` longtext NOT NULL,
  `exception` longtext NOT NULL,
  `failed_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `failed_jobs_uuid_unique` (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `failed_jobs` */

/*Table structure for table `migrations` */

DROP TABLE IF EXISTS `migrations`;

CREATE TABLE `migrations` (
  `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `migration` varchar(255) NOT NULL,
  `batch` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `migrations` */

insert  into `migrations`(`id`,`migration`,`batch`) values 
(1,'2014_10_12_000000_create_users_table',1),
(2,'2014_10_12_100000_create_password_reset_tokens_table',1),
(3,'2019_08_19_000000_create_failed_jobs_table',1),
(4,'2019_12_14_000001_create_personal_access_tokens_table',1),
(5,'2023_12_14_124922_create_orders_table',2),
(6,'2023_12_14_125301_create_products_table',2),
(7,'2023_12_14_130053_add_product_id_to_orders_table',3),
(8,'2023_12_17_080201_change_description_to_text',4),
(9,'2023_12_17_091818_modify_orders_table',5),
(10,'2023_12_17_092323_rename_order_number_column_in_orders_table',6),
(11,'2023_12_17_092651_add_completed_column_to_orders_table',7),
(12,'2023_12_17_093250_create_order_product_table',8),
(13,'2023_12_17_093912_remove_order_uuid_column_from_orders_table',8),
(14,'2023_12_17_210851_add_url_to_products_table',9),
(15,'2023_12_18_112708_create_brands_table',10),
(16,'2023_12_18_113053_add_image_to_brands_table',11);

/*Table structure for table `order_product` */

DROP TABLE IF EXISTS `order_product`;

CREATE TABLE `order_product` (
  `order_id` bigint(20) unsigned NOT NULL,
  `product_id` bigint(20) unsigned NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`order_id`,`product_id`),
  KEY `order_product_product_id_foreign` (`product_id`),
  CONSTRAINT `order_product_order_id_foreign` FOREIGN KEY (`order_id`) REFERENCES `orders` (`id`) ON DELETE CASCADE,
  CONSTRAINT `order_product_product_id_foreign` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `order_product` */

/*Table structure for table `orders` */

DROP TABLE IF EXISTS `orders`;

CREATE TABLE `orders` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint(20) unsigned NOT NULL,
  `completed` tinyint(1) NOT NULL DEFAULT 0,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `orders_user_id_foreign` (`user_id`),
  CONSTRAINT `orders_user_id_foreign` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `orders` */

/*Table structure for table `password_reset_tokens` */

DROP TABLE IF EXISTS `password_reset_tokens`;

CREATE TABLE `password_reset_tokens` (
  `email` varchar(255) NOT NULL,
  `token` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `password_reset_tokens` */

/*Table structure for table `personal_access_tokens` */

DROP TABLE IF EXISTS `personal_access_tokens`;

CREATE TABLE `personal_access_tokens` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `tokenable_type` varchar(255) NOT NULL,
  `tokenable_id` bigint(20) unsigned NOT NULL,
  `name` varchar(255) NOT NULL,
  `token` varchar(64) NOT NULL,
  `abilities` text DEFAULT NULL,
  `last_used_at` timestamp NULL DEFAULT NULL,
  `expires_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `personal_access_tokens_token_unique` (`token`),
  KEY `personal_access_tokens_tokenable_type_tokenable_id_index` (`tokenable_type`,`tokenable_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `personal_access_tokens` */

/*Table structure for table `products` */

DROP TABLE IF EXISTS `products`;

CREATE TABLE `products` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` text NOT NULL,
  `url` varchar(255) DEFAULT NULL,
  `price` decimal(10,2) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `products` */

insert  into `products`(`id`,`name`,`description`,`url`,`price`,`created_at`,`updated_at`) values 
(1,'Lew Robel','Incidunt ipsam itaque ut. Nisi error distinctio vel eum expedita ut. Et odio voluptate ratione minus architecto vero aliquid. Libero est est vel atque eaque. Aut vitae maxime dolores quis.',NULL,65.13,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(2,'Miss Velma Aufderhar PhD','Dicta quis ex adipisci magnam sit quae qui sit. Voluptas maxime necessitatibus perspiciatis. Ad fugit delectus itaque enim asperiores dignissimos ad. Atque nam rem quia rerum ex rem est veniam. Aut deserunt ad aut. Explicabo veritatis vero quia eligendi. Inventore repellendus tempora asperiores dolor veritatis minima.',NULL,34.13,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(3,'Robb Schoen','Consequatur mollitia ullam ullam est et et. Dolore porro ea cupiditate doloribus nesciunt. Quia error fugiat commodi et laudantium in. Ea voluptatem quia inventore molestias molestiae et aliquid. Maxime quasi praesentium sapiente non.',NULL,3.25,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(4,'Virginie Walter','Commodi et ullam ullam eum sequi aut qui. Non nam sed iusto nemo velit. Earum molestiae ut facere et sint nulla. Ut rerum debitis consequuntur aut harum dolor accusantium. Aut aspernatur iste sit modi sed omnis. Magni ipsam molestiae est necessitatibus sed optio esse. Reprehenderit aut sed eum vel repellendus aut.',NULL,11.45,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(5,'Brandon Kautzer Sr.','Eum exercitationem facilis quasi sapiente ullam quo sit inventore. Et aut qui vel dignissimos excepturi velit. Sint accusamus ut eligendi accusantium. Quibusdam pariatur quia est sit nihil itaque veritatis. Assumenda fuga voluptas et non voluptates voluptatem. Libero pariatur est omnis facilis delectus numquam atque.',NULL,68.16,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(6,'Carmen Ruecker','Omnis quo velit fuga dolorum placeat voluptatem. Ad omnis eius quod eos labore facilis numquam. Possimus voluptate harum enim tempora qui eum doloribus unde. Natus dolores deleniti ab qui atque.',NULL,49.32,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(7,'Merl Kshlerin','Quasi consequatur ipsam quia quasi non saepe. Earum ab est et quo aut. Natus quia suscipit non facere odit. Harum neque velit alias recusandae enim.',NULL,28.60,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(8,'Angie Kirlin','Nemo accusantium repellat quos culpa. Dolores dolore nemo laboriosam laborum qui omnis mollitia. Et autem eos et expedita. Dolorem et sint corporis aliquid perferendis.',NULL,3.73,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(9,'Aileen Powlowski','Animi earum et saepe et dolor. Vitae vero ducimus quo ullam enim deleniti eum perspiciatis. Mollitia placeat velit sunt tempore corrupti nemo. Nihil fugiat sed minus quam nihil error. Nesciunt deserunt aperiam eum facilis voluptatum aliquam ipsam. Accusamus non rem id molestiae.',NULL,6.16,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(10,'Eino Kutch','Nobis suscipit deserunt fugiat amet laudantium nostrum consequuntur. Doloremque dolore sed enim. Iure doloremque qui quidem voluptate. Sequi sunt cum quis sed repellat.',NULL,2.12,'2023-12-17 08:23:06','2023-12-17 08:23:06');

/*Table structure for table `users` */

DROP TABLE IF EXISTS `users`;

CREATE TABLE `users` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `email_verified_at` timestamp NULL DEFAULT NULL,
  `password` varchar(255) NOT NULL,
  `remember_token` varchar(100) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  `roles` varchar(255) DEFAULT 'USER',
  PRIMARY KEY (`id`),
  UNIQUE KEY `users_email_unique` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `users` */

insert  into `users`(`id`,`name`,`email`,`email_verified_at`,`password`,`remember_token`,`created_at`,`updated_at`,`roles`) values 
(1,'Milind','7milind7@gmail.com',NULL,'$2y$12$AFA2HdkE1PcXxpk9w4raaeJyvX/p5zfDZNynnvSOuHEgDskY5lqRi','Z2qG7124p3WoPA7Aq7O9U4E3DdxSpUmC86gjtSXBCS08XRoHqisI7Y675DUM','2023-12-13 08:38:50','2023-12-13 08:38:50','USER'),
(2,'Admin','admin@localhost.com',NULL,'$2y$12$arO629x1nODz5XdnwpIlQu/uVuQOlUN1n/w6jJ7j9CJgS5KTbPA..','gr78cwasXSlJfbylvwf4t4bFghar5XxCxkoqp9Qrx395lBsW22EO29MnvLzb','2023-12-13 09:08:52','2023-12-13 09:08:52','ADMIN'),
(3,'Vergie Anderson','loy63@example.net','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','xaFPKfWBsQ','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(4,'Dr. Paxton Volkman','camylle.brakus@example.net','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','mql1qQfGtU','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(5,'Devonte Hoppe','audra.barrows@example.com','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','q0Bub4CmJ6','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(6,'Delores Kemmer','pcollier@example.com','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','BVzgiCa18j','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(7,'Emmitt Zboncak','cullen.reynolds@example.net','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','l6RTwumzSc','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(8,'Antonina Rice','rowan16@example.com','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','KQGyvP3Hfl','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(9,'Dr. Mariane Lind','block.justus@example.net','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','P6ml3SKrN5','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(10,'Maribel Harvey','ohills@example.com','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','UZj7N0wqQK','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(11,'Alvis Johnston','leslie.johns@example.org','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','C3AfeAmamQ','2023-12-17 09:14:38','2023-12-17 09:14:38','USER'),
(12,'Ms. Karlie Rau V','garrison63@example.org','2023-12-17 09:14:38','$2y$12$Qt8VGi43wVhQhflTztY.uO7H6dIjJJSb8P2g8303qTyjhYYi.EeF6','lxsFldIobU','2023-12-17 09:14:38','2023-12-17 09:14:38','USER');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
