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
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
(11,'2023_12_17_092651_add_completed_column_to_orders_table',7);

/*Table structure for table `orders` */

DROP TABLE IF EXISTS `orders`;

CREATE TABLE `orders` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `order_uuid` char(36) NOT NULL,
  `user_id` bigint(20) unsigned NOT NULL,
  `completed` tinyint(1) NOT NULL DEFAULT 0,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `orders_order_number_unique` (`order_uuid`),
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
  `price` decimal(10,2) NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

/*Data for the table `products` */

insert  into `products`(`id`,`name`,`description`,`price`,`created_at`,`updated_at`) values 
(1,'Lew Robel','Incidunt ipsam itaque ut. Nisi error distinctio vel eum expedita ut. Et odio voluptate ratione minus architecto vero aliquid. Libero est est vel atque eaque. Aut vitae maxime dolores quis.',65.13,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(2,'Miss Velma Aufderhar PhD','Dicta quis ex adipisci magnam sit quae qui sit. Voluptas maxime necessitatibus perspiciatis. Ad fugit delectus itaque enim asperiores dignissimos ad. Atque nam rem quia rerum ex rem est veniam. Aut deserunt ad aut. Explicabo veritatis vero quia eligendi. Inventore repellendus tempora asperiores dolor veritatis minima.',34.13,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(3,'Robb Schoen','Consequatur mollitia ullam ullam est et et. Dolore porro ea cupiditate doloribus nesciunt. Quia error fugiat commodi et laudantium in. Ea voluptatem quia inventore molestias molestiae et aliquid. Maxime quasi praesentium sapiente non.',3.25,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(4,'Virginie Walter','Commodi et ullam ullam eum sequi aut qui. Non nam sed iusto nemo velit. Earum molestiae ut facere et sint nulla. Ut rerum debitis consequuntur aut harum dolor accusantium. Aut aspernatur iste sit modi sed omnis. Magni ipsam molestiae est necessitatibus sed optio esse. Reprehenderit aut sed eum vel repellendus aut.',11.45,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(5,'Brandon Kautzer Sr.','Eum exercitationem facilis quasi sapiente ullam quo sit inventore. Et aut qui vel dignissimos excepturi velit. Sint accusamus ut eligendi accusantium. Quibusdam pariatur quia est sit nihil itaque veritatis. Assumenda fuga voluptas et non voluptates voluptatem. Libero pariatur est omnis facilis delectus numquam atque.',68.16,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(6,'Carmen Ruecker','Omnis quo velit fuga dolorum placeat voluptatem. Ad omnis eius quod eos labore facilis numquam. Possimus voluptate harum enim tempora qui eum doloribus unde. Natus dolores deleniti ab qui atque.',49.32,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(7,'Merl Kshlerin','Quasi consequatur ipsam quia quasi non saepe. Earum ab est et quo aut. Natus quia suscipit non facere odit. Harum neque velit alias recusandae enim.',28.60,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(8,'Angie Kirlin','Nemo accusantium repellat quos culpa. Dolores dolore nemo laboriosam laborum qui omnis mollitia. Et autem eos et expedita. Dolorem et sint corporis aliquid perferendis.',3.73,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(9,'Aileen Powlowski','Animi earum et saepe et dolor. Vitae vero ducimus quo ullam enim deleniti eum perspiciatis. Mollitia placeat velit sunt tempore corrupti nemo. Nihil fugiat sed minus quam nihil error. Nesciunt deserunt aperiam eum facilis voluptatum aliquam ipsam. Accusamus non rem id molestiae.',6.16,'2023-12-17 08:23:06','2023-12-17 08:23:06'),
(10,'Eino Kutch','Nobis suscipit deserunt fugiat amet laudantium nostrum consequuntur. Doloremque dolore sed enim. Iure doloremque qui quidem voluptate. Sequi sunt cum quis sed repellat.',2.12,'2023-12-17 08:23:06','2023-12-17 08:23:06');

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
